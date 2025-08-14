# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import ttnn
from loguru import logger

from models.experimental.SSR.tt.atten_blocks import TTAttenBlocks
from models.experimental.SSR.tests.test_HAB import create_hab_preprocessor, create_relative_position_index
from models.experimental.SSR.tests.test_OCAB import create_ocab_preprocessor
from models.utility_functions import comp_pcc


def create_atten_blocks_preprocessor(device, depth):
    """Preprocessor for AttenBlocks that handles multiple HAB blocks and one OCAB block"""

    def custom_preprocessor(torch_model, name, ttnn_module_args):
        params = {}

        # Preprocess parameters for each HAB block
        params["blocks"] = {}
        hab_preprocessor = create_hab_preprocessor(device)
        for i in range(depth):
            params["blocks"][i] = hab_preprocessor(torch_model.blocks[i], f"blocks_{i}", ttnn_module_args)

        # Preprocess parameters for OCAB
        ocab_preprocessor = create_ocab_preprocessor(device)
        params["overlap_attn"] = ocab_preprocessor(torch_model.overlap_attn, "overlap_attn")

        return params

    return custom_preprocessor


@pytest.mark.parametrize(
    "batch_size, height, width, dim, num_heads, window_size, depth, overlap_ratio, mlp_ratio",
    [
        (1, 64, 64, 180, 6, 16, 2, 0.5, 2.0),  # Standard configuration
        (1, 32, 32, 96, 3, 8, 3, 0.25, 4.0),  # Smaller resolution, more blocks
        (2, 64, 64, 180, 6, 16, 1, 0.5, 2.0),  # Batch size 2, single block
        (1, 128, 128, 192, 6, 16, 2, 0.75, 3.0),  # Larger resolution
        (2, 64, 64, 180, 6, 16, 6, 0.5, 2),  # Network config
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_atten_blocks(device, batch_size, height, width, dim, num_heads, window_size, depth, overlap_ratio, mlp_ratio):
    torch.manual_seed(0)

    # Import reference model
    from models.experimental.SSR.reference.SSR.model.tile_refinement import AttenBlocks

    # Create reference model
    ref_model = AttenBlocks(
        dim=dim,
        input_resolution=(height, width),
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=overlap_ratio,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    )
    ref_model.eval()

    # Create input tensors
    input_tensor = torch.randn(batch_size, height * width, dim)
    x_size = (height, width)

    # Create relative position indices
    rpi_sa = create_relative_position_index((window_size, window_size))

    # Create attention mask for shifted windows (simplified for testing)
    attn_mask = None

    # Create RPI for OCAB
    overlap_win_size = int(window_size * overlap_ratio) + window_size
    rpi_oca = torch.zeros((window_size * window_size, overlap_win_size * overlap_win_size), dtype=torch.long)

    # Create params dictionary
    params = {"rpi_sa": rpi_sa, "attn_mask": attn_mask, "rpi_oca": rpi_oca}

    # Reference forward pass
    with torch.no_grad():
        ref_output = ref_model(input_tensor, x_size, params)

    # Create TTNN model
    parameters = ttnn.model_preprocessing.preprocess_model(
        initialize_model=lambda: ref_model,
        custom_preprocessor=create_atten_blocks_preprocessor(device, depth),
        device=device,
        run_model=lambda model: model(input_tensor, x_size, params),
    )

    tt_model = TTAttenBlocks(
        device=device,
        parameters=parameters,
        dim=dim,
        input_resolution=(height, width),
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=overlap_ratio,
        mlp_ratio=mlp_ratio,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Convert inputs to TTNN format
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    tt_rpi_sa = ttnn.from_torch(rpi_sa, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)

    tt_rpi_oca = ttnn.from_torch(rpi_oca, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32)

    tt_params = {"rpi_sa": tt_rpi_sa, "attn_mask": None, "rpi_oca": tt_rpi_oca}

    # TTNN forward pass
    tt_output = tt_model(tt_input, x_size, tt_params)

    # Convert back to PyTorch format
    tt_torch_output = ttnn.to_torch(tt_output)

    # Compare outputs
    does_pass, pcc_message = comp_pcc(ref_output, tt_torch_output, 0.90)

    logger.info(f"Batch: {batch_size}, Size: {height}x{width}, Dim: {dim}")
    logger.info(f"Heads: {num_heads}, Window: {window_size}, Depth: {depth}")
    logger.info(f"Overlap ratio: {overlap_ratio}, MLP ratio: {mlp_ratio}")
    logger.info(f"Reference output shape: {ref_output.shape}")
    logger.info(f"TTNN output shape: {tt_torch_output.shape}")
    logger.info(pcc_message)

    if does_pass:
        logger.info("AttenBlocks Passed!")
    else:
        logger.warning("AttenBlocks Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
    assert (
        ref_output.shape == tt_torch_output.shape
    ), f"Shape mismatch: ref {ref_output.shape} vs ttnn {tt_torch_output.shape}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_atten_blocks_memory_config(device):
    """Test different memory configurations"""
    torch.manual_seed(0)

    from models.experimental.SSR.reference.SSR.model.tile_refinement import AttenBlocks

    dim = 96
    height, width = 32, 32
    num_heads = 3
    window_size = 8
    depth = 2
    overlap_ratio = 0.5

    ref_model = AttenBlocks(
        dim=dim,
        input_resolution=(height, width),
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=overlap_ratio,
        mlp_ratio=4.0,
    )
    ref_model.eval()

    input_tensor = torch.randn(1, height * width, dim)
    x_size = (height, width)
    rpi_sa = create_relative_position_index((window_size, window_size))
    overlap_win_size = int(window_size * overlap_ratio) + window_size
    rpi_oca = torch.zeros((window_size * window_size, overlap_win_size * overlap_win_size), dtype=torch.long)

    params = {"rpi_sa": rpi_sa, "attn_mask": None, "rpi_oca": rpi_oca}

    with torch.no_grad():
        ref_output = ref_model(input_tensor, x_size, params)

    # Test with DRAM memory config
    parameters = ttnn.model_preprocessing.preprocess_model(
        initialize_model=lambda: ref_model,
        custom_preprocessor=create_atten_blocks_preprocessor(device, depth),
        device=device,
        run_model=lambda model: model(input_tensor, x_size, params),
    )

    tt_model_dram = TTAttenBlocks(
        device=device,
        parameters=parameters,
        dim=dim,
        input_resolution=(height, width),
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=overlap_ratio,
        mlp_ratio=4.0,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_rpi_sa = ttnn.from_torch(rpi_sa, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)
    tt_rpi_oca = ttnn.from_torch(rpi_oca, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32)

    tt_params = {"rpi_sa": tt_rpi_sa, "attn_mask": None, "rpi_oca": tt_rpi_oca}

    tt_output_dram = tt_model_dram(tt_input, x_size, tt_params)
    tt_torch_output_dram = ttnn.to_torch(tt_output_dram)

    does_pass_dram, pcc_message_dram = comp_pcc(ref_output, tt_torch_output_dram, 0.90)
    logger.info(f"DRAM Memory Config: {pcc_message_dram}")
    assert does_pass_dram, f"DRAM memory config failed: {pcc_message_dram}"

    # Clean up tensors
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output_dram)

    logger.info("AttenBlocks Memory Config Test Passed!")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_atten_blocks_multiple_iterations(device):
    """Test multiple forward passes to check for memory leaks"""
    torch.manual_seed(0)

    from models.experimental.SSR.reference.SSR.model.tile_refinement import AttenBlocks

    dim = 96
    height, width = 64, 64
    num_heads = 3
    window_size = 8
    depth = 2
    overlap_ratio = 0.5

    ref_model = AttenBlocks(
        dim=dim,
        input_resolution=(height, width),
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=overlap_ratio,
        mlp_ratio=4.0,
    )
    ref_model.eval()

    parameters = ttnn.model_preprocessing.preprocess_model(
        initialize_model=lambda: ref_model,
        custom_preprocessor=create_atten_blocks_preprocessor(device, depth),
        device=device,
        run_model=lambda model: model(
            torch.randn(1, height * width, dim),
            (height, width),
            {
                "rpi_sa": create_relative_position_index((window_size, window_size)),
                "attn_mask": None,
                "rpi_oca": torch.zeros(
                    (window_size * window_size, (int(window_size * overlap_ratio) + window_size) ** 2), dtype=torch.long
                ),
            },
        ),
    )

    tt_model = TTAttenBlocks(
        device=device,
        parameters=parameters,
        dim=dim,
        input_resolution=(height, width),
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=overlap_ratio,
        mlp_ratio=4.0,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run multiple forward passes to check for memory leaks
    for i in range(3):
        input_tensor = torch.randn(1, height * width, dim)
        x_size = (height, width)
        rpi_sa = create_relative_position_index((window_size, window_size))
        overlap_win_size = int(window_size * overlap_ratio) + window_size
        rpi_oca = torch.zeros((window_size * window_size, overlap_win_size * overlap_win_size), dtype=torch.long)

        params = {"rpi_sa": rpi_sa, "attn_mask": None, "rpi_oca": rpi_oca}

        with torch.no_grad():
            ref_output = ref_model(input_tensor, x_size, params)

        tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tt_rpi_sa = ttnn.from_torch(rpi_sa, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)
        tt_rpi_oca = ttnn.from_torch(rpi_oca, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32)

        tt_params = {"rpi_sa": tt_rpi_sa, "attn_mask": None, "rpi_oca": tt_rpi_oca}

        tt_output = tt_model(tt_input, x_size, tt_params)
        tt_torch_output = ttnn.to_torch(tt_output)

        does_pass, pcc_message = comp_pcc(ref_output, tt_torch_output, 0.90)
        logger.info(f"Iteration {i+1}: {pcc_message}")
        assert does_pass, f"Iteration {i+1} failed: {pcc_message}"

        # Clean up tensors to prevent memory leaks
        ttnn.deallocate(tt_input)
        ttnn.deallocate(tt_rpi_sa)
        ttnn.deallocate(tt_rpi_oca)
        ttnn.deallocate(tt_output)

        logger.info(f"Iteration {i+1} completed successfully")

    logger.info("AttenBlocks Multiple Iterations Test Passed!")
