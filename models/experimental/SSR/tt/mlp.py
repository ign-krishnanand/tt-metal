import ttnn

from models.common.lightweightmodule import LightweightModule


class TTMlp(LightweightModule):
    def __init__(self, device, memory_config, in_features, hidden_features=None, out_features=None, ref_layer=None):
        self.memory_config = memory_config if memory_config is not None else ttnn.L1_MEMORY_CONFIG
        self.device = device

        # self.fc1_weight = None  # Shape: [hidden_features, in_features]
        # self.fc1_bias = None    # Shape: [hidden_features]
        # self.fc2_weight = None  # Shape: [out_features, hidden_features]
        # self.fc2_bias = None

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        # self._initialize_random_weights()

        self.fc1_weight = ttnn.from_torch(ref_layer.fc1.weight.permute(1, 0), device=device, layout=ttnn.TILE_LAYOUT)
        self.fc1_bias = ttnn.from_torch(ref_layer.fc1.bias, device=device, layout=ttnn.TILE_LAYOUT)

        self.fc2_weight = ttnn.from_torch(ref_layer.fc2.weight.permute(1, 0), device=device, layout=ttnn.TILE_LAYOUT)
        self.fc2_bias = ttnn.from_torch(ref_layer.fc2.bias, device=device, layout=ttnn.TILE_LAYOUT)

    def _initialize_random_weights(self):
        # Create random weights for fc1: [in_features, hidden_features]
        self.fc1_weight = ttnn.rand(
            [self.in_features, self.hidden_features],
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
        )

        # Create random bias for fc1: [1, hidden_features]
        self.fc1_bias = ttnn.rand(
            [1, self.hidden_features],
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
        )

        # Create random weights for fc2: [hidden_features, out_features]
        self.fc2_weight = ttnn.rand(
            [self.hidden_features, self.out_features],
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
        )

        # Create random bias for fc2: [1, out_features]
        self.fc2_bias = ttnn.rand(
            [1, self.out_features],
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
        )

    def forward(self, x):
        x = ttnn.linear(x, self.fc1_weight, bias=self.fc1_bias, memory_config=self.memory_config)

        # Activation function
        x = ttnn.gelu(x)

        # Second linear layer
        x = ttnn.linear(x, self.fc2_weight, bias=self.fc2_bias, memory_config=self.memory_config)

        return x
