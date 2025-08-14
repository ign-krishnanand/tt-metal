# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import ttnn
from loguru import logger

from models.experimental.SSR.tt.HAB import TTWindowAttention
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight
from models.utility_functions import comp_pcc


class WindowAttention(nn.Module):
    """Reference PyTorch WindowAttention implementation"""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def create_window_attention_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        params = {}

        # QKV linear layer
        params["qkv"] = {
            "weight": preprocess_linear_weight(torch_model.qkv.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            "bias": preprocess_linear_bias(torch_model.qkv.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            if torch_model.qkv.bias is not None
            else None,
        }

        # Projection layer
        params["proj"] = {
            "weight": preprocess_linear_weight(torch_model.proj.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            "bias": preprocess_linear_bias(torch_model.proj.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            if torch_model.proj.bias is not None
            else None,
        }

        # Relative position bias table
        params["relative_position_bias_table"] = ttnn.from_torch(
            torch_model.relative_position_bias_table, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )

        return params

    return custom_preprocessor


@pytest.mark.parametrize(
    "batch_size, num_windows, window_size, dim, num_heads",
    [
        # (1, 4, (7, 7), 96, 3),      # Standard configuration
        # (1, 16, (7, 7), 192, 6),    # Larger feature dimension
        # (2, 4, (7, 7), 96, 3),      # Batch size 2
        # (1, 9, (8, 8), 128, 4),     # Different window size
        # (1, 1, (14, 14), 384, 12),  # Large window
        (1, 1, (16, 14), 180, 6),  # Large window
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_window_attention(device, batch_size, num_windows, window_size, dim, num_heads):
    torch.manual_seed(0)

    # Create reference model
    ref_model = WindowAttention(
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    )
    ref_model.eval()

    # Create input tensors
    window_area = window_size[0] * window_size[1]
    input_tensor = torch.randn(batch_size * num_windows, window_area, dim)

    # Create relative position index
    coords_h = torch.arange(window_size[0])
    coords_w = torch.arange(window_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size[0] - 1
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    rpi = relative_coords.sum(-1)

    # Reference forward pass
    with torch.no_grad():
        ref_output = ref_model(input_tensor, rpi=rpi, mask=None)

    # Create TTNN model
    parameters = ttnn.model_preprocessing.preprocess_model(
        initialize_model=lambda: ref_model,
        custom_preprocessor=create_window_attention_preprocessor(device),
        device=device,
        run_model=lambda model: model(input_tensor, rpi=rpi, mask=None),
    )

    tt_model = TTWindowAttention(
        device=device,
        parameters=parameters,
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Convert inputs to TTNN format
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    tt_rpi = ttnn.from_torch(rpi, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)

    # TTNN forward pass
    tt_output = tt_model(tt_input, rpi=tt_rpi, mask=None)

    # Convert back to PyTorch format
    tt_torch_output = ttnn.to_torch(tt_output)

    # Compare outputs
    does_pass, pcc_message = comp_pcc(ref_output, tt_torch_output, 0.97)

    logger.info(f"Batch: {batch_size}, Windows: {num_windows}, Window size: {window_size}")
    logger.info(f"Dim: {dim}, Heads: {num_heads}")
    logger.info(f"Reference output shape: {ref_output.shape}")
    logger.info(f"TTNN output shape: {tt_torch_output.shape}")
    logger.info(pcc_message)

    if does_pass:
        logger.info("Window Attention Passed!")
    else:
        logger.warning("Window Attention Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
    assert (
        ref_output.shape == tt_torch_output.shape
    ), f"Shape mismatch: ref {ref_output.shape} vs ttnn {tt_torch_output.shape}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_window_attention_with_mask(device):
    """Test window attention with attention mask"""
    torch.manual_seed(0)

    dim = 96
    window_size = (7, 7)
    num_heads = 3
    batch_size = 1
    num_windows = 4

    ref_model = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads)
    ref_model.eval()

    window_area = window_size[0] * window_size[1]
    input_tensor = torch.randn(batch_size * num_windows, window_area, dim)

    # Create relative position index
    coords_h = torch.arange(window_size[0])
    coords_w = torch.arange(window_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size[0] - 1
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    rpi = relative_coords.sum(-1)

    # Create attention mask
    mask = torch.zeros(num_windows, window_area, window_area)
    mask[:, : window_area // 2, window_area // 2 :] = float("-inf")  # Mask half the attention

    with torch.no_grad():
        ref_output = ref_model(input_tensor, rpi=rpi, mask=mask)

    parameters = ttnn.model_preprocessing.preprocess_model(
        initialize_model=lambda: ref_model,
        custom_preprocessor=create_window_attention_preprocessor(device),
        device=device,
        run_model=lambda model: model(input_tensor, rpi=rpi, mask=mask),
    )

    tt_model = TTWindowAttention(
        device=device,
        parameters=parameters,
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_rpi = ttnn.from_torch(rpi, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)
    tt_mask = ttnn.from_torch(mask, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    tt_output = tt_model(tt_input, rpi=tt_rpi, mask=tt_mask)
    tt_torch_output = ttnn.to_torch(tt_output)

    does_pass, pcc_message = comp_pcc(ref_output, tt_torch_output, 0.95)
    logger.info(f"Window Attention with Mask: {pcc_message}")
    assert does_pass, f"Masked attention test failed: {pcc_message}"
