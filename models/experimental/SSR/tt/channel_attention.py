# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule


class TTChannelAttention(LightweightModule):
    def __init__(self, device, parameters, num_feat, squeeze_factor=16, memory_config=None):
        super().__init__()
        self.device = device
        self.memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
        self.num_feat = num_feat
        self.squeeze_factor = squeeze_factor

        # Extract preprocessed parameters
        self.conv1_weight = parameters["conv1"]["weight"]
        self.conv1_bias = parameters["conv1"]["bias"]
        self.conv2_weight = parameters["conv2"]["weight"]
        self.conv2_bias = parameters["conv2"]["bias"]

    def forward(self, x):
        # Store original input for multiplication
        original_x = x
        original_shape = x.shape

        # Global Average Pooling (AdaptiveAvgPool2d(1) equivalent)
        x = ttnn.global_avg_pool2d(x, memory_config=self.memory_config)

        # First 1x1 convolution (squeeze)
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv1_weight,
            bias_tensor=self.conv1_bias,
            device=self.device,
            in_channels=self.num_feat,
            out_channels=self.num_feat // self.squeeze_factor,
            batch_size=x.shape[0],
            input_height=1,
            input_width=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            memory_config=self.memory_config,
        )

        # ReLU activation
        x = ttnn.relu(x)

        # Second 1x1 convolution (excitation)
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv2_weight,
            bias_tensor=self.conv2_bias,
            device=self.device,
            in_channels=self.num_feat // self.squeeze_factor,
            out_channels=self.num_feat,
            batch_size=x.shape[0],
            input_height=1,
            input_width=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            memory_config=self.memory_config,
        )

        # Sigmoid activation
        x = ttnn.sigmoid(x)

        batch_size, height, width, channels = original_shape
        attention_weights = ttnn.reshape(x, [batch_size, 1, 1, channels])
        attention_weights = ttnn.repeat(attention_weights, [1, height, width, 1], memory_config=self.memory_config)

        # Element-wise multiplication with original input
        output = ttnn.multiply(original_x, attention_weights, memory_config=self.memory_config)

        return output
