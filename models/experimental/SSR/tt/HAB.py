import ttnn
from models.common.lightweightmodule import LightweightModule

from models.experimental.SSR.tt.CAB import TTCAB
from models.experimental.SSR.tt.mlp import TTMlp
from models.experimental.SSR.tt.window_attn_tr import TTWindowAttentionTR


class TTHAB(LightweightModule):
    def __init__(
        self,
        device,
        parameters,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        memory_config=None,
    ):
        super().__init__()
        self.device = device
        self.memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Adjust window size and shift size if needed
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        # Extract preprocessed parameters
        self.norm1_weight = parameters["norm1"]["weight"]
        self.norm1_bias = parameters["norm1"]["bias"]
        self.norm2_weight = parameters["norm2"]["weight"]
        self.norm2_bias = parameters["norm2"]["bias"]
        self.conv_scale = parameters.get("conv_scale", 0.01)

        # Initialize sub-modules
        self.attn = TTWindowAttentionTR(
            device=device,
            parameters=parameters["attn"],
            dim=dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            memory_config=memory_config,
        )

        self.conv_block = TTCAB(
            device=device, parameters=parameters["conv_block"], num_feat=dim, memory_config=memory_config
        )

        self.mlp = TTMlp(
            device=device,
            memory_config=memory_config,
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            parameters=parameters["mlp"],
        )

    def forward(self, x, x_size, rpi_sa, attn_mask):
        h, w = x_size
        b, seq_len, c = x.shape
        shortcut = x

        # Layer norm 1
        x = ttnn.layer_norm(x, weight=self.norm1_weight, bias=self.norm1_bias)

        # Reshape to spatial format for conv and attention
        x = ttnn.reshape(x, [b, h, w, c])

        # Convolutional branch
        conv_x = self.conv_block(x)
        conv_x = ttnn.reshape(conv_x, [b, h * w, c])
        conv_x = ttnn.multiply(conv_x, self.conv_scale)

        # Attention branch - handle cyclic shift
        if self.shift_size > 0:
            # Cyclic shift
            shifted_x = ttnn.roll(x, [-self.shift_size, -self.shift_size], [1, 2])
            current_attn_mask = attn_mask
        else:
            shifted_x = x
            current_attn_mask = None

        # Window partition
        x_windows = self._window_partition(shifted_x, self.window_size)
        x_windows = ttnn.reshape(x_windows, [-1, self.window_size * self.window_size, c])

        # Window attention
        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=current_attn_mask)

        # Window reverse
        attn_windows = ttnn.reshape(attn_windows, [-1, self.window_size, self.window_size, c])
        shifted_x = self._window_reverse(attn_windows, self.window_size, h, w)

        # Reverse cyclic shift
        if self.shift_size > 0:
            attn_x = ttnn.roll(shifted_x, [self.shift_size, self.shift_size], [1, 2])
        else:
            attn_x = shifted_x

        attn_x = ttnn.reshape(attn_x, [b, h * w, c])

        # First residual connection
        x = ttnn.add(shortcut, attn_x)
        x = ttnn.add(x, conv_x)

        # MLP branch
        x_norm = ttnn.layer_norm(x, weight=self.norm2_weight, bias=self.norm2_bias)
        mlp_out = self.mlp(x_norm)

        # Second residual connection
        x = ttnn.add(x, mlp_out)

        return x

    def _window_partition(self, x, window_size):
        """Partition into non-overlapping windows"""
        b, h, w, c = x.shape
        x = ttnn.reshape(x, [b, h // window_size, window_size, w // window_size, window_size, c])
        x = ttnn.permute(x, [0, 1, 3, 2, 4, 5])
        windows = ttnn.reshape(x, [-1, window_size, window_size, c])
        return windows

    def _window_reverse(self, windows, window_size, h, w):
        """Reverse window partition"""
        b = windows.shape[0] // (h * w // window_size // window_size)
        x = ttnn.reshape(windows, [b, h // window_size, w // window_size, window_size, window_size, -1])
        x = ttnn.permute(x, [0, 1, 3, 2, 4, 5])
        x = ttnn.reshape(x, [b, h, w, -1])
        return x
