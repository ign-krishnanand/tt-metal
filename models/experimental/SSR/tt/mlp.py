import ttnn
from models.common.lightweightmodule import LightweightModule


class TTMlp(LightweightModule):
    def __init__(self, device, memory_config, in_features, hidden_features=None, out_features=None, parameters=None):
        self.memory_config = memory_config if memory_config is not None else ttnn.L1_MEMORY_CONFIG
        self.device = device

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        # Initialize weights and biases based on available inputs
        # Use preprocessed parameters
        self.fc1_weight = parameters["fc1"]["weight"]
        self.fc1_bias = parameters["fc1"]["bias"]
        self.fc2_weight = parameters["fc2"]["weight"]
        self.fc2_bias = parameters["fc2"]["bias"]

    def forward(self, x):
        x = ttnn.linear(x, self.fc1_weight, bias=self.fc1_bias, memory_config=self.memory_config)

        # Activation function
        x = ttnn.gelu(x)

        # Second linear layer
        x = ttnn.linear(x, self.fc2_weight, bias=self.fc2_bias, memory_config=self.memory_config)

        return x
