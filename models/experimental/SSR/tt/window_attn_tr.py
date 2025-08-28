import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3.utils.config_helpers import matmul_config


class TTWindowAttentionTR(LightweightModule):
    def __init__(self, device, parameters, dim, window_size, num_heads, memory_config=None):
        super().__init__()
        self.device = device
        self.memory_config = memory_config or ttnn.L1_MEMORY_CONFIG
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Extract preprocessed parameters
        self.qkv_weight = parameters["qkv"]["weight"]
        self.qkv_bias = parameters["qkv"]["bias"] if "bias" in parameters["qkv"] else None
        self.proj_weight = parameters["proj"]["weight"]
        self.proj_bias = parameters["proj"]["bias"] if "bias" in parameters["proj"] else None
        self.relative_position_bias_table = parameters["relative_position_bias_table"]

        # Scale factor
        self.scale = self.head_dim**-0.5

    def forward(self, x, rpi, mask=None):
        b_, n, c = x.shape
        # QKV projection
        # qkv_program_config = matmul_config(x.shape[-2], x.shape[-1], self.qkv_bias.shape[-1], (8, 8))
        # qkv = ttnn.linear(x, self.qkv_weight, bias=self.qkv_bias, memory_config=self.memory_config, program_config=qkv_program_config)
        # ttnn.deallocate(x)

        # x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        qkv = ttnn.linear(
            x,
            self.qkv_weight,
            bias=self.qkv_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG if b_ * n * c < 1_100_000 else ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=ttnn.CoreGrid(y=7, x=7),
        )
        ttnn.deallocate(x)
        # unoptimised method - works for 180 dim
        qkv = ttnn.reshape(
            qkv,
            [b_, n, 3, self.num_heads, self.head_dim],
            memory_config=ttnn.L1_MEMORY_CONFIG if b_ * n * c < 1_100_000 else ttnn.DRAM_MEMORY_CONFIG,
        )
        qkv = ttnn.permute(qkv, [2, 0, 3, 1, 4])  # [3, b_, num_heads, n, head_dim]

        # Split Q, K, V
        q = ttnn.slice(qkv, [0, 0, 0, 0, 0], [1, b_, self.num_heads, n, self.head_dim])
        k = ttnn.slice(qkv, [1, 0, 0, 0, 0], [2, b_, self.num_heads, n, self.head_dim])
        v = ttnn.slice(qkv, [2, 0, 0, 0, 0], [3, b_, self.num_heads, n, self.head_dim])

        # -------------------------------------------------------------

        # optimised method - Works for 192 dim
        # padding = [(0, 0), (0, 0), (0, 576 - 540)]
        # qkv = ttnn.pad(qkv, padding, 0.0)
        # # import pdb; pdb.set_trace()
        # (
        #     q,
        #     k,
        #     v,
        # ) = ttnn.transformer.split_query_key_value_and_split_heads(qkv,memory_config=ttnn.L1_MEMORY_CONFIG,num_heads=self.num_heads)
        # # Deallocate the original qkv tensor
        # ttnn.deallocate(qkv)

        # # import pdb; pdb.set_trace()
        # q = q[:, :, :, :30]
        # k = k[:, :, :30, :]
        # v = v[:, :, :, :30]
        # # -------------------------------------------------------------

        # Remove the first dimension
        q = ttnn.squeeze(q, 0)
        k = ttnn.squeeze(k, 0)
        v = ttnn.squeeze(v, 0)

        # Scale Q
        q = ttnn.multiply(q, self.scale, memory_config=self.memory_config)

        # Attention computation: Q @ K^T
        k_transposed = ttnn.transpose(
            k, -2, -1, memory_config=self.memory_config
        )  # not required in the optimised method
        attn = ttnn.matmul(
            q,
            k_transposed,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG if b_ * n * c < 1_100_000 else ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(k_transposed)

        # Add relative position bias
        # Extract relative position bias from table using rpi indices
        window_area = self.window_size[0] * self.window_size[1]

        rpi_flat = ttnn.reshape(rpi, [-1])
        relative_position_bias = ttnn.embedding(
            rpi_flat, self.relative_position_bias_table, memory_config=self.memory_config
        )
        relative_position_bias = ttnn.reshape(
            relative_position_bias, [window_area, window_area, self.num_heads], memory_config=self.memory_config
        )
        relative_position_bias = ttnn.permute(
            relative_position_bias, [2, 0, 1], memory_config=self.memory_config
        )  # [num_heads, window_area, window_area]

        # Add bias to attention
        relative_position_bias = ttnn.unsqueeze(relative_position_bias, 0)  # [1, num_heads, window_area, window_area]
        attn = ttnn.add(attn, relative_position_bias, memory_config=self.memory_config)

        # Apply mask if provided
        if mask is not None:
            nw = mask.shape[0]
            attn = ttnn.reshape(attn, [b_ // nw, nw, self.num_heads, n, n])
            mask_expanded = ttnn.unsqueeze(ttnn.unsqueeze(mask, 1), 0)
            attn = ttnn.add(attn, mask_expanded)
            attn = ttnn.reshape(attn, [-1, self.num_heads, n, n])

        # Softmax
        attn = ttnn.softmax(attn, dim=-1, memory_config=self.memory_config)

        # Apply attention to values
        x = ttnn.matmul(
            attn,
            v,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=self.memory_config,
        )  # [b_, num_heads, n, head_dim]

        # Transpose and reshape back
        x = ttnn.transpose(x, 1, 2, memory_config=self.memory_config)  # [b_, n, num_heads, head_dim]
        # import pdb; pdb.set_trace()
        x = ttnn.reshape(x, [b_, n, c], memory_config=self.memory_config)
        # x = ttnn.reshape(x, [b_, n, 192], memory_config=self.memory_config)
        # import pdb; pdb.set_trace()
        # x = x[:, :, :180]
        # Output projection
        # program_config = matmul_config(
        #     x.shape[-2], x.shape[-1], self.proj_bias.shape[-1], (7, 7)
        # )
        # import pdb; pdb.set_trace()
        program_config = matmul_config(x.shape[-2], x.shape[-1], self.proj_bias.shape[-1], (8, 8))
        x = ttnn.linear(
            x, self.proj_weight, bias=self.proj_bias, memory_config=self.memory_config, program_config=program_config
        )
        return x
