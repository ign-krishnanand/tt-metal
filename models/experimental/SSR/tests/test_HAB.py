# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.SSR.tt.HAB import TTHAB
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight
from models.utility_functions import comp_pcc


# def create_hab_preprocessor(device):
#     def custom_preprocessor(torch_model, name, ttnn_module_args):
#         params = {}

#         # Norm layers
#         params["norm1"] = {
#             "weight": preprocess_linear_weight(torch_model.norm1.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
#             "bias": preprocess_linear_bias(torch_model.norm1.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
#         }
#         params["norm2"] = {
#             "weight": preprocess_linear_weight(torch_model.norm2.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
#             "bias": preprocess_linear_bias(torch_model.norm2.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
#         }

#         # Window attention parameters
#         params["attn"] = {
#             "qkv": {
#                 "weight": preprocess_linear_weight(torch_model.attn.qkv.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
#                 "bias": preprocess_linear_bias(torch_model.attn.qkv.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) if torch_model.attn.qkv.bias is not None else None
#             },
#             "proj": {
#                 "weight": preprocess_linear_weight(torch_model.attn.proj.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
#                 "bias": preprocess_linear_bias(torch_model.attn.proj.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) if torch_model.attn.proj.bias is not None else None
#             },
#             "relative_position_bias_table": ttnn.from_torch(
#                 torch_model.attn.relative_position_bias_table, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
#             )
#         }

#         # Conv block (CAB) parameters
#         cab_layers = list(torch_model.conv_block.cab.children())
#         conv1 = cab_layers[0]
#         conv2 = cab_layers[2]
#         channel_attention = cab_layers[3]

#         params["conv_block"] = {
#             "conv1": {
#                 "weight": ttnn.from_torch(conv1.weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
#                 "bias": ttnn.from_torch(conv1.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
#             },
#             "conv2": {
#                 "weight": ttnn.from_torch(conv2.weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
#                 "bias": ttnn.from_torch(conv2.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
#             },
#             "channel_attention": {
#                 "avg_pool_weight": ttnn.from_torch(channel_attention.avg_pool.weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
#                 "avg_pool_bias": ttnn.from_torch(channel_attention.avg_pool.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
#                 "conv_du_weight": ttnn.from_torch(channel_attention.conv_du.weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
#                 "conv_du_bias": ttnn.from_torch(channel_attention.conv_du.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
#             }
#         }

#         # MLP parameters
#         params["mlp"] = {
#             "fc1": {
#                 "weight": preprocess_linear_weight(torch_model.mlp.fc1.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
#                 "bias": preprocess_linear_bias(torch_model.mlp.fc1.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) if torch_model.mlp.fc1.bias is not None else None
#             },
#             "fc2": {
#                 "weight": preprocess_linear_weight(torch_model.mlp.fc2.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
#                 "bias": preprocess_linear_bias(torch_model.mlp.fc2.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) if torch_model.mlp.fc2.bias is not None else None
#             }
#         }

#         # Conv scale
#         params["conv_scale"] = torch_model.conv_scale

#         return params

#     return custom_preprocessor


def create_hab_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        params = {}

        # Norm layers
        params["norm1"] = {
            "weight": preprocess_linear_weight(torch_model.norm1.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            "bias": preprocess_linear_bias(torch_model.norm1.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        }
        params["norm2"] = {
            "weight": preprocess_linear_weight(torch_model.norm2.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            "bias": preprocess_linear_bias(torch_model.norm2.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        }

        # Window attention parameters
        params["attn"] = {
            "qkv": {
                "weight": preprocess_linear_weight(
                    torch_model.attn.qkv.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                ),
                "bias": preprocess_linear_bias(torch_model.attn.qkv.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                if torch_model.attn.qkv.bias is not None
                else None,
            },
            "proj": {
                "weight": preprocess_linear_weight(
                    torch_model.attn.proj.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                ),
                "bias": preprocess_linear_bias(torch_model.attn.proj.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                if torch_model.attn.proj.bias is not None
                else None,
            },
            "relative_position_bias_table": ttnn.from_torch(
                torch_model.attn.relative_position_bias_table, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            ),
        }

        # Conv block (CAB) parameters - Fixed to match your channel attention structure
        cab_layers = list(torch_model.conv_block.cab.children())
        conv1 = cab_layers[0]  # First Conv2d layer
        conv2 = cab_layers[2]  # Second Conv2d layer (after GELU)
        channel_attention = cab_layers[3]  # ChannelAttention module

        # Extract the sequential layers from ChannelAttention (matching your test pattern)
        attention_layers = list(channel_attention.attention.children())
        attention_conv1 = attention_layers[1]  # First Conv2d layer in attention
        attention_conv2 = attention_layers[3]  # Second Conv2d layer in attention

        conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)

        params["conv_block"] = {
            "conv1": {
                "weight": ttnn.from_torch(conv1.weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
                "bias": ttnn.from_torch(
                    conv1.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
                ),
            },
            "conv2": {
                "weight": ttnn.from_torch(conv2.weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
                "bias": ttnn.from_torch(
                    conv2.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
                ),
            },
            "channel_attention": {
                "conv1": {
                    "weight": ttnn.prepare_conv_weights(
                        weight_tensor=ttnn.from_torch(attention_conv1.weight, dtype=ttnn.bfloat16),
                        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        input_layout=ttnn.TILE_LAYOUT,
                        weights_format="OIHW",
                        in_channels=attention_conv1.in_channels,
                        out_channels=attention_conv1.out_channels,
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
                        bias_tensor=ttnn.from_torch(attention_conv1.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16),
                        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        input_layout=ttnn.TILE_LAYOUT,
                        in_channels=attention_conv1.in_channels,
                        out_channels=attention_conv1.out_channels,
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
                },
                "conv2": {
                    "weight": ttnn.prepare_conv_weights(
                        weight_tensor=ttnn.from_torch(attention_conv2.weight, dtype=ttnn.bfloat16),
                        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        input_layout=ttnn.TILE_LAYOUT,
                        weights_format="OIHW",
                        in_channels=attention_conv2.in_channels,
                        out_channels=attention_conv2.out_channels,
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
                        bias_tensor=ttnn.from_torch(attention_conv2.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16),
                        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        input_layout=ttnn.TILE_LAYOUT,
                        in_channels=attention_conv2.in_channels,
                        out_channels=attention_conv2.out_channels,
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
                },
            },
        }

        # MLP parameters
        params["mlp"] = {
            "fc1": {
                "weight": preprocess_linear_weight(
                    torch_model.mlp.fc1.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                ),
                "bias": preprocess_linear_bias(torch_model.mlp.fc1.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                if torch_model.mlp.fc1.bias is not None
                else None,
            },
            "fc2": {
                "weight": preprocess_linear_weight(
                    torch_model.mlp.fc2.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                ),
                "bias": preprocess_linear_bias(torch_model.mlp.fc2.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                if torch_model.mlp.fc2.bias is not None
                else None,
            },
        }

        # Conv scale
        params["conv_scale"] = torch_model.conv_scale

        return params

    return custom_preprocessor


def create_relative_position_index(window_size):
    """Create relative position index for window attention"""
    coords_h = torch.arange(window_size[0])
    coords_w = torch.arange(window_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size[0] - 1
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    return relative_coords.sum(-1)


@pytest.mark.parametrize(
    "batch_size, height, width, dim, num_heads, window_size, shift_size, mlp_ratio",
    [
        # (1, 32, 32, 96, 3, 8, 0, 4.0),  # Standard configuration
        # (1, 64, 64, 192, 6, 8, 0, 4.0),  # Larger feature dimension
        # (2, 32, 32, 96, 3, 8, 0, 4.0),  # Batch size 2
        # (1, 56, 56, 128, 4, 7, 0, 4.0),  # Different window size
        # (1, 32, 32, 180, 6, 8, 4, 4.0),  # With shift
        (2, 64, 64, 180, 6, 16, 8, 2.0),  # With shift
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_hab_block(device, batch_size, height, width, dim, num_heads, window_size, shift_size, mlp_ratio):
    torch.manual_seed(0)

    # Import reference model here to avoid dependency issues
    from models.experimental.SSR.reference.SSR.model.tile_refinement import HAB

    # Create reference model
    ref_model = HAB(
        dim=dim,
        input_resolution=(height, width),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
    )
    ref_model.eval()

    # Create input tensors
    input_tensor = torch.randn(batch_size, height * width, dim)
    x_size = (height, width)

    # Create relative position index
    rpi_sa = create_relative_position_index((window_size, window_size))

    # Create attention mask for shifted windows
    if shift_size > 0:
        img_mask = torch.zeros((1, height, width, 1))
        h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # Create attention mask
        mask_windows = img_mask.view(1, height // window_size, window_size, width // window_size, window_size, 1)
        mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf")).masked_fill(attn_mask == 0, float("0.0"))
    else:
        attn_mask = None

    # Reference forward pass
    with torch.no_grad():
        ref_output = ref_model(input_tensor, x_size, rpi_sa, attn_mask)

    # Create TTNN model
    parameters = ttnn.model_preprocessing.preprocess_model(
        initialize_model=lambda: ref_model,
        custom_preprocessor=create_hab_preprocessor(device),
        device=device,
        run_model=lambda model: model(input_tensor, x_size, rpi_sa, attn_mask),
    )

    tt_model = TTHAB(
        device=device,
        parameters=parameters,
        dim=dim,
        input_resolution=(height, width),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        mlp_ratio=mlp_ratio,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Convert inputs to TTNN format
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    tt_rpi = ttnn.from_torch(rpi_sa, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)

    tt_attn_mask = None
    if attn_mask is not None:
        tt_attn_mask = ttnn.from_torch(attn_mask, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # TTNN forward pass
    tt_output = tt_model(tt_input, x_size, tt_rpi, tt_attn_mask)

    # Convert back to PyTorch format
    tt_torch_output = ttnn.to_torch(tt_output)

    # Compare outputs
    does_pass, pcc_message = comp_pcc(ref_output, tt_torch_output, 0.95)

    logger.info(f"Batch: {batch_size}, Size: {height}x{width}, Dim: {dim}")
    logger.info(f"Heads: {num_heads}, Window: {window_size}, Shift: {shift_size}")
    logger.info(f"Reference output shape: {ref_output.shape}")
    logger.info(f"TTNN output shape: {tt_torch_output.shape}")
    logger.info(pcc_message)

    if does_pass:
        logger.info("HAB Block Passed!")
    else:
        logger.warning("HAB Block Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
    assert (
        ref_output.shape == tt_torch_output.shape
    ), f"Shape mismatch: ref {ref_output.shape} vs ttnn {tt_torch_output.shape}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_hab_memory_config(device):
    """Test different memory configurations"""
    torch.manual_seed(0)

    from models.experimental.SSR.reference.SSR.model.tile_refinement import HAB

    dim = 96
    height, width = 32, 32
    num_heads = 3
    window_size = 8

    ref_model = HAB(
        dim=dim,
        input_resolution=(height, width),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=0,
        mlp_ratio=4.0,
    )
    ref_model.eval()

    input_tensor = torch.randn(1, height * width, dim)
    x_size = (height, width)
    rpi_sa = create_relative_position_index((window_size, window_size))

    with torch.no_grad():
        ref_output = ref_model(input_tensor, x_size, rpi_sa, None)

    # Test with L1 memory config
    parameters = ttnn.model_preprocessing.preprocess_model(
        initialize_model=lambda: ref_model,
        custom_preprocessor=create_hab_preprocessor(device),
        device=device,
        run_model=lambda model: model(input_tensor, x_size, rpi_sa, None),
    )

    tt_model_l1 = TTHAB(
        device=device,
        parameters=parameters,
        dim=dim,
        input_resolution=(height, width),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=0,
        mlp_ratio=4.0,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    tt_rpi = ttnn.from_torch(rpi_sa, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)

    tt_output_l1 = tt_model_l1(tt_input, x_size, tt_rpi, None)
    tt_torch_output_l1 = ttnn.to_torch(tt_output_l1)

    does_pass_l1, pcc_message_l1 = comp_pcc(ref_output, tt_torch_output_l1, 0.95)
    logger.info(f"L1 Memory Config: {pcc_message_l1}")
    assert does_pass_l1, f"L1 memory config failed: {pcc_message_l1}"

    # Test with DRAM memory config
    tt_model_dram = TTHAB(
        device=device,
        parameters=parameters,
        dim=dim,
        input_resolution=(height, width),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=0,
        mlp_ratio=4.0,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output_dram = tt_model_dram(tt_input, x_size, tt_rpi, None)
    tt_torch_output_dram = ttnn.to_torch(tt_output_dram)

    does_pass_dram, pcc_message_dram = comp_pcc(ref_output, tt_torch_output_dram, 0.95)
    logger.info(f"DRAM Memory Config: {pcc_message_dram}")
    assert does_pass_dram, f"DRAM memory config failed: {pcc_message_dram}"

    # Clean up tensors
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output_l1)
    ttnn.deallocate(tt_output_dram)

    logger.info("HAB Memory Config Test Passed!")


@pytest.mark.parametrize(
    "compress_ratio, squeeze_factor",
    [
        (2, 16),  # High compression
        (4, 32),  # Low compression
        (3, 8),  # High squeeze
        (6, 64),  # Low squeeze
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_hab_compression_ratios(device, compress_ratio, squeeze_factor):
    """Test different compression and squeeze factor configurations"""
    torch.manual_seed(0)

    pytest.xfail("Only one of 'min' or 'max' can be None. Please provide one value")

    from models.experimental.SSR.reference.SSR.model.tile_refinement import HAB

    dim = 96
    height, width = 32, 32
    num_heads = 3
    window_size = 8

    ref_model = HAB(
        dim=dim,
        input_resolution=(height, width),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=0,
        mlp_ratio=4.0,
        compress_ratio=compress_ratio,
        squeeze_factor=squeeze_factor,
    )
    ref_model.eval()

    input_tensor = torch.randn(1, height * width, dim)
    x_size = (height, width)
    rpi_sa = create_relative_position_index((window_size, window_size))

    with torch.no_grad():
        ref_output = ref_model(input_tensor, x_size, rpi_sa, None)

    parameters = ttnn.model_preprocessing.preprocess_model(
        initialize_model=lambda: ref_model,
        custom_preprocessor=create_hab_preprocessor(device),
        device=device,
        run_model=lambda model: model(input_tensor, x_size, rpi_sa, None),
    )

    tt_model = TTHAB(
        device=device,
        parameters=parameters,
        dim=dim,
        input_resolution=(height, width),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=0,
        mlp_ratio=4.0,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    tt_rpi = ttnn.from_torch(rpi_sa, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)

    tt_output = tt_model(tt_input, x_size, tt_rpi, None)
    tt_torch_output = ttnn.to_torch(tt_output)

    does_pass, pcc_message = comp_pcc(ref_output, tt_torch_output, 0.95)

    logger.info(f"Compress ratio: {compress_ratio}, Squeeze factor: {squeeze_factor}")
    logger.info(f"Reference output shape: {ref_output.shape}")
    logger.info(f"TTNN output shape: {tt_torch_output.shape}")
    logger.info(pcc_message)

    if does_pass:
        logger.info("HAB Compression Test Passed!")
    else:
        logger.warning("HAB Compression Test Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"

    # Clean up tensors
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_hab_multiple_iterations(device):
    """Test multiple forward passes to check for memory leaks"""
    torch.manual_seed(0)

    from models.experimental.SSR.reference.SSR.model.tile_refinement import HAB

    dim = 128
    height, width = 64, 64
    num_heads = 4
    window_size = 8

    ref_model = HAB(
        dim=dim,
        input_resolution=(height, width),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=0,
        mlp_ratio=4.0,
    )
    ref_model.eval()

    parameters = ttnn.model_preprocessing.preprocess_model(
        initialize_model=lambda: ref_model,
        custom_preprocessor=create_hab_preprocessor(device),
        device=device,
        run_model=lambda model: model(
            torch.randn(1, height * width, dim),
            (height, width),
            create_relative_position_index((window_size, window_size)),
            None,
        ),
    )

    tt_model = TTHAB(
        device=device,
        parameters=parameters,
        dim=dim,
        input_resolution=(height, width),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=0,
        mlp_ratio=4.0,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run multiple forward passes to check for memory leaks
    for i in range(3):
        input_tensor = torch.randn(1, height * width, dim)
        x_size = (height, width)
        rpi_sa = create_relative_position_index((window_size, window_size))

        with torch.no_grad():
            ref_output = ref_model(input_tensor, x_size, rpi_sa, None)

        tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

        tt_rpi = ttnn.from_torch(rpi_sa, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)

        tt_output = tt_model(tt_input, x_size, tt_rpi, None)
        tt_torch_output = ttnn.to_torch(tt_output)

        does_pass, pcc_message = comp_pcc(ref_output, tt_torch_output, 0.95)
        logger.info(f"Iteration {i+1}: {pcc_message}")
        assert does_pass, f"Iteration {i+1} failed: {pcc_message}"

        # Clean up tensors to prevent memory leaks
        ttnn.deallocate(tt_input)
        ttnn.deallocate(tt_output)

        logger.info(f"Iteration {i+1} completed successfully")
