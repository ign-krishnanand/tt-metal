# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.SSR.tt.channel_attention import TTChannelAttention
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import comp_pcc

from models.experimental.SSR.reference.SSR.model.tile_refinement import ChannelAttention


# class ChannelAttention(nn.Module):
#     """Reference PyTorch ChannelAttention implementation"""
#     def __init__(self, num_feat, squeeze_factor=16):
#         super(ChannelAttention, self).__init__()
#         self.attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         y = self.attention(x)
#         return x * y


def create_channel_attention_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        params = {}

        # Extract the sequential layers
        layers = list(torch_model.attention.children())
        conv1 = layers[1]  # First Conv2d layer
        conv2 = layers[3]  # Second Conv2d layer

        conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)

        # Preprocess first convolution
        params["conv1"] = {
            "weight": ttnn.prepare_conv_weights(
                weight_tensor=ttnn.from_torch(conv1.weight, dtype=ttnn.bfloat16),
                input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                input_layout=ttnn.TILE_LAYOUT,
                weights_format="OIHW",
                in_channels=conv1.in_channels,
                out_channels=conv1.out_channels,
                batch_size=1,
                input_height=1,
                input_width=1,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                has_bias=True,
                groups=1,
                device=device,
                input_dtype=ttnn.bfloat16,
                conv_config=conv_config,
            ),
            "bias": ttnn.prepare_conv_bias(
                bias_tensor=ttnn.from_torch(
                    conv1.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16  # Reshape to 4D: [1, 1, 1, out_channels]
                ),
                input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                input_layout=ttnn.TILE_LAYOUT,
                in_channels=conv1.in_channels,
                out_channels=conv1.out_channels,
                batch_size=1,
                input_height=1,
                input_width=1,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
                device=device,
                input_dtype=ttnn.bfloat16,
                conv_config=conv_config,
            ),
        }

        # Preprocess second convolution
        params["conv2"] = {
            "weight": ttnn.prepare_conv_weights(
                weight_tensor=ttnn.from_torch(conv2.weight, dtype=ttnn.bfloat16),
                input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                input_layout=ttnn.TILE_LAYOUT,
                weights_format="OIHW",
                in_channels=conv2.in_channels,
                out_channels=conv2.out_channels,
                batch_size=1,
                input_height=1,
                input_width=1,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                has_bias=True,
                groups=1,
                device=device,
                input_dtype=ttnn.bfloat16,
                conv_config=conv_config,
            ),
            "bias": ttnn.prepare_conv_bias(
                bias_tensor=ttnn.from_torch(
                    conv2.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16  # Reshape to 4D: [1, 1, 1, out_channels]
                ),
                input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                input_layout=ttnn.TILE_LAYOUT,
                in_channels=conv2.in_channels,
                out_channels=conv2.out_channels,
                batch_size=1,
                input_height=1,
                input_width=1,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
                device=device,
                input_dtype=ttnn.bfloat16,
                conv_config=conv_config,
            ),
        }

        return params

    return custom_preprocessor


@pytest.mark.parametrize(
    "batch_size, num_feat, height, width, squeeze_factor",
    [
        (2, 64, 32, 32, 16),  # Standard RCAN configuration
        (1, 128, 64, 64, 16),  # Larger feature maps
        (2, 64, 32, 32, 16),  # Batch size 2
        (1, 256, 16, 16, 32),  # Higher channels with different squeeze factor
        (1, 32, 128, 128, 8),  # Smaller channels, larger spatial dimensions
        (1, 512, 8, 8, 64),  # Very high channels, small spatial
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_channel_attention(device, batch_size, num_feat, height, width, squeeze_factor):
    torch.manual_seed(0)

    # Create reference model
    ref_model = ChannelAttention(num_feat=num_feat, squeeze_factor=squeeze_factor)
    ref_model.eval()

    # Create input tensor
    input_tensor = torch.randn(batch_size, num_feat, height, width)

    # Reference forward pass
    with torch.no_grad():
        ref_output = ref_model(input_tensor)

    # Create TTNN model
    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        custom_preprocessor=create_channel_attention_preprocessor(device),
        device=device,
    )

    tt_model = TTChannelAttention(
        device=device,
        parameters=parameters,
        num_feat=num_feat,
        squeeze_factor=squeeze_factor,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Convert input to TTNN format (NHWC)
    tt_input = ttnn.from_torch(
        input_tensor.permute(0, 2, 3, 1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16  # NCHW -> NHWC
    )

    # TTNN forward pass
    tt_output = tt_model(tt_input)

    # Convert back to PyTorch format
    tt_torch_output = ttnn.to_torch(tt_output)
    tt_torch_output = tt_torch_output.permute(0, 3, 1, 2)  # NHWC -> NCHW

    # Compare outputs
    does_pass, pcc_message = comp_pcc(ref_output, tt_torch_output, 0.98)

    logger.info(f"Batch: {batch_size}, Features: {num_feat}, Size: {height}x{width}, Squeeze: {squeeze_factor}")
    logger.info(f"Reference output shape: {ref_output.shape}")
    logger.info(f"TTNN output shape: {tt_torch_output.shape}")
    logger.info(pcc_message)

    if does_pass:
        logger.info("ChannelAttention Passed!")
    else:
        logger.warning("ChannelAttention Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"

    # Verify output shapes match
    assert (
        ref_output.shape == tt_torch_output.shape
    ), f"Shape mismatch: ref {ref_output.shape} vs ttnn {tt_torch_output.shape}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_channel_attention_memory_config(device):
    """Test different memory configurations"""
    torch.manual_seed(0)

    num_feat = 64
    squeeze_factor = 16
    batch_size = 1
    height, width = 32, 32

    ref_model = ChannelAttention(num_feat=num_feat, squeeze_factor=squeeze_factor)
    ref_model.eval()

    input_tensor = torch.randn(batch_size, num_feat, height, width)

    with torch.no_grad():
        ref_output = ref_model(input_tensor)

    # Test with L1 memory config
    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        custom_preprocessor=create_channel_attention_preprocessor(device),
        device=device,
    )

    tt_model_l1 = TTChannelAttention(
        device=device,
        parameters=parameters,
        num_feat=num_feat,
        squeeze_factor=squeeze_factor,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tt_input = ttnn.from_torch(
        input_tensor.permute(0, 2, 3, 1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    tt_output_l1 = tt_model_l1(tt_input)
    tt_torch_output_l1 = ttnn.to_torch(tt_output_l1).permute(0, 3, 1, 2)

    does_pass_l1, pcc_message_l1 = comp_pcc(ref_output, tt_torch_output_l1, 0.98)
    logger.info(f"L1 Memory Config: {pcc_message_l1}")
    assert does_pass_l1, f"L1 memory config failed: {pcc_message_l1}"


def test_channel_attention_multiple_iterations(device):
    """Test multiple forward passes to check for memory leaks"""
    torch.manual_seed(0)

    num_feat = 128
    squeeze_factor = 16
    batch_size = 1
    height, width = 64, 64

    ref_model = ChannelAttention(num_feat=num_feat, squeeze_factor=squeeze_factor)
    ref_model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        custom_preprocessor=create_channel_attention_preprocessor(device),
        device=device,
    )

    tt_model = TTChannelAttention(
        device=device,
        parameters=parameters,
        num_feat=num_feat,
        squeeze_factor=squeeze_factor,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run multiple forward passes to check for memory leaks
    for i in range(3):
        input_tensor = torch.randn(batch_size, num_feat, height, width)

        with torch.no_grad():
            ref_output = ref_model(input_tensor)

        tt_input = ttnn.from_torch(
            input_tensor.permute(0, 2, 3, 1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

        tt_output = tt_model(tt_input)
        tt_torch_output = ttnn.to_torch(tt_output).permute(0, 3, 1, 2)

        does_pass, pcc_message = comp_pcc(ref_output, tt_torch_output, 0.98)
        logger.info(f"Iteration {i+1}: {pcc_message}")
        assert does_pass, f"Iteration {i+1} failed: {pcc_message}"

        # Clean up tensors to prevent memory leaks
        ttnn.deallocate(tt_input)
        ttnn.deallocate(tt_output)

        logger.info(f"Iteration {i+1} completed successfully")


@pytest.mark.parametrize(
    "num_feat, squeeze_factor",
    [
        (64, 4),  # High squeeze factor
        (128, 64),  # Low squeeze factor
        (256, 16),  # Standard squeeze factor
        (32, 2),  # Minimal squeeze factor
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_channel_attention_squeeze_factors(device, num_feat, squeeze_factor):
    """Test different squeeze factor configurations"""
    torch.manual_seed(0)

    batch_size = 1
    height, width = 32, 32

    ref_model = ChannelAttention(num_feat=num_feat, squeeze_factor=squeeze_factor)
    ref_model.eval()

    input_tensor = torch.randn(batch_size, num_feat, height, width)

    with torch.no_grad():
        ref_output = ref_model(input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        custom_preprocessor=create_channel_attention_preprocessor(device),
        device=device,
    )

    tt_model = TTChannelAttention(
        device=device,
        parameters=parameters,
        num_feat=num_feat,
        squeeze_factor=squeeze_factor,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_input = ttnn.from_torch(
        input_tensor.permute(0, 2, 3, 1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    tt_output = tt_model(tt_input)
    tt_torch_output = ttnn.to_torch(tt_output).permute(0, 3, 1, 2)

    does_pass, pcc_message = comp_pcc(ref_output, tt_torch_output, 0.98)

    logger.info(f"Reference output shape: {ref_output.shape}")
    logger.info(f"TTNN output shape: {tt_torch_output.shape}")
    logger.info(pcc_message)

    if does_pass:
        logger.info("ChannelAttention Passed!")
    else:
        logger.warning("ChannelAttention Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"

    # Verify output shapes match
    assert (
        ref_output.shape == tt_torch_output.shape
    ), f"Shape mismatch: ref {ref_output.shape} vs ttnn {tt_torch_output.shape}"
