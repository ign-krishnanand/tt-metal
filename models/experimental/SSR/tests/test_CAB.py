# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.  
# SPDX-License-Identifier: Apache-2.0  
  
import pytest  
import torch  
import torch.nn as nn  
import ttnn  
from loguru import logger  
  
from models.experimental.SSR.tt.CAB import TTCAB  
from ttnn.model_preprocessing import preprocess_model_parameters  
from models.utility_functions import comp_pcc  
from models.experimental.SSR.reference.SSR.model.tile_refinement import ChannelAttention  
from models.experimental.SSR.tests.test_channel_attention import create_channel_attention_preprocessor  
  
  
class CAB(nn.Module):  
    """Reference PyTorch CAB implementation"""  
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):  
        super(CAB, self).__init__()  
  
        self.cab = nn.Sequential(  
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),  
            nn.GELU(),  
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),  
            ChannelAttention(num_feat, squeeze_factor),  
        )  
  
    def forward(self, x):  
        return self.cab(x)  
  
  
def create_cab_preprocessor(device):  
    def custom_preprocessor(torch_model, name, ttnn_module_args):  
        params = {}  
          
        # Extract the sequential layers from CAB  
        cab_layers = list(torch_model.cab.children())  
        conv1 = cab_layers[0]  # First Conv2d layer  
        conv2 = cab_layers[2]  # Second Conv2d layer (after GELU)  
        channel_attention = cab_layers[3]  # ChannelAttention module  
          
        # Create conv config  
        conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)  

        actual_height = 32 #ttnn_module_args.get('height', 32)  # Use actual height  
        actual_width = 32 #ttnn_module_args.get('width', 32)

          
        # Preprocess first convolution (3x3)  
        params["conv1"] = {  
            "weight": ttnn.from_torch(conv1.weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
            # ttnn.prepare_conv_weights(  
            #     weight_tensor=ttnn.from_torch(conv1.weight, dtype=ttnn.bfloat16),  
            #     input_memory_config=ttnn.DRAM_MEMORY_CONFIG,  
            #     input_layout=ttnn.TILE_LAYOUT,  
            #     weights_format="OIHW",  
            #     in_channels=conv1.in_channels,  
            #     out_channels=conv1.out_channels,  
            #     batch_size=1,  
            #     input_height=1,  
            #     input_width=1,  
            #     kernel_size=(3, 3),  
            #     stride=(1, 1),  
            #     padding=(1, 1),  
            #     dilation=(1, 1),  
            #     has_bias=True,  
            #     groups=1,  
            #     device=device,  
            #     input_dtype=ttnn.bfloat16,  
            #     conv_config=conv_config,  
            # ),  
            "bias": ttnn.from_torch(  
                    conv1.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
                ), 
            # ttnn.prepare_conv_bias(  
            #     bias_tensor=ttnn.from_torch(  
            #         conv1.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16  
            #     ),  
            #     input_memory_config=ttnn.DRAM_MEMORY_CONFIG,  
            #     input_layout=ttnn.TILE_LAYOUT,  
            #     in_channels=conv1.in_channels,  
            #     out_channels=conv1.out_channels,  
            #     batch_size=1,  
            #     input_height=48,  
            #     input_width=48, 
            #     kernel_size=(3, 3),  
            #     stride=(1, 1),  
            #     padding=(1, 1),  
            #     dilation=(1, 1),  
            #     groups=1,  
            #     device=device,  
            #     input_dtype=ttnn.bfloat16,  
            #     conv_config=conv_config,  
            # )  
        } 
          
        # Preprocess second convolution (3x3)  
        params["conv2"] = {  
            "weight": ttnn.from_torch(conv2.weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT), 
            # ttnn.prepare_conv_weights(  
            #     weight_tensor=ttnn.from_torch(conv2.weight, dtype=ttnn.bfloat16),  
            #     input_memory_config=ttnn.DRAM_MEMORY_CONFIG,  
            #     input_layout=ttnn.TILE_LAYOUT,  
            #     weights_format="OIHW",  
            #     in_channels=conv2.in_channels,  
            #     out_channels=conv2.out_channels,  
            #     batch_size=1,  
            #     input_height=actual_height,  
            #     input_width=actual_width, 
            #     kernel_size=(3, 3),  
            #     stride=(1, 1),  
            #     padding=(1, 1),  
            #     dilation=(1, 1),  
            #     has_bias=True,  
            #     groups=1,  
            #     device=device,  
            #     input_dtype=ttnn.bfloat16,  
            #     conv_config=conv_config,  
            # ),  
            "bias": ttnn.from_torch(  
                    conv2.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT 
                ),  
            # ttnn.prepare_conv_bias(  
            #     bias_tensor=ttnn.from_torch(  
            #         conv2.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16  
            #     ),  
            #     input_memory_config=ttnn.DRAM_MEMORY_CONFIG,  
            #     input_layout=ttnn.TILE_LAYOUT,  
            #     in_channels=conv2.in_channels,  
            #     out_channels=conv2.out_channels,  
            #     batch_size=1,  
            #     input_height=actual_height,  
            #     input_width=actual_width,  
            #     kernel_size=(3, 3),  
            #     stride=(1, 1),  
            #     padding=(1, 1),  
            #     dilation=(1, 1),  
            #     groups=1,  
            #     device=device,  
            #     input_dtype=ttnn.bfloat16,  
            #     conv_config=conv_config,  
            # )  
        }  
          
        # Preprocess channel attention using existing preprocessor  
        channel_attention_preprocessor = create_channel_attention_preprocessor(device)  
        params["channel_attention"] = channel_attention_preprocessor(channel_attention, "channel_attention", ttnn_module_args)  
          
        return params  
      
    return custom_preprocessor  
  
  
@pytest.mark.parametrize(  
    "batch_size, num_feat, height, width, compress_ratio, squeeze_factor",  
    [  
        (1, 64, 24, 24, 3, 30),      # Standard configuration  
        (1, 64, 32, 32, 3, 30),      # Standard configuration  
        (1, 128, 64, 64, 4, 16),     # Larger feature maps  
        (2, 64, 32, 32, 3, 30),      # Batch size 2  
        (1, 96, 48, 48, 2, 24),      # Different compress ratio  
        (1, 32, 128, 128, 3, 8),     # Smaller channels, larger spatial  
        (1, 256, 16, 16, 8, 64),     # High channels, small spatial  
    ],  
)  
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)  
def test_cab_block(device, batch_size, num_feat, height, width, compress_ratio, squeeze_factor):  
    torch.manual_seed(0)  
  
    # Create reference model  
    ref_model = CAB(num_feat=num_feat, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)  
    ref_model.eval()  
  
    # Create input tensor  
    input_tensor = torch.randn(batch_size, num_feat, height, width)  
  
    # Reference forward pass  
    with torch.no_grad():  
        ref_output = ref_model(input_tensor)  
  
    # Create TTNN model  
    # parameters = preprocess_model_parameters(  
    #     initialize_model=lambda: ref_model,  
    #     custom_preprocessor=create_cab_preprocessor(device),  
    #     device=device,  
    # )  

    parameters = ttnn.model_preprocessing.preprocess_model(  
        initialize_model=lambda: ref_model,  
        custom_preprocessor=create_cab_preprocessor(device),  
        device=device,  
        run_model=lambda model: model(input_tensor),  
    ) 
  
    tt_model = TTCAB(  
        device=device,  
        parameters=parameters,  
        num_feat=num_feat,  
        compress_ratio=compress_ratio,  
        squeeze_factor=squeeze_factor,  
        memory_config=ttnn.DRAM_MEMORY_CONFIG,  
    )  
  
    # Convert input to TTNN format (NHWC)  
    tt_input = ttnn.from_torch(  
        input_tensor.permute(0, 2, 3, 1),  # NCHW -> NHWC  
        device=device,  
        layout=ttnn.TILE_LAYOUT,  
        dtype=ttnn.bfloat16  
    )  
  
    # TTNN forward pass  
    tt_output = tt_model(tt_input)  
  
    # Convert back to PyTorch format  
    tt_torch_output = ttnn.to_torch(tt_output)  
    tt_torch_output = tt_torch_output.permute(0, 3, 1, 2)  # NHWC -> NCHW  
  
    # Compare outputs  
    does_pass, pcc_message = comp_pcc(ref_output, tt_torch_output, 0.97)  
  
    logger.info(f"Batch: {batch_size}, Features: {num_feat}, Size: {height}x{width}")  
    logger.info(f"Compress ratio: {compress_ratio}, Squeeze factor: {squeeze_factor}")  
    logger.info(f"Reference output shape: {ref_output.shape}")  
    logger.info(f"TTNN output shape: {tt_torch_output.shape}")  
    logger.info(pcc_message)  
  
    if does_pass:  
        logger.info("CAB Block Passed!")  
    else:  
        logger.warning("CAB Block Failed!")  
  
    assert does_pass, f"PCC check failed: {pcc_message}"  
  
    # Verify output shapes match  
    assert ref_output.shape == tt_torch_output.shape, f"Shape mismatch: ref {ref_output.shape} vs ttnn {tt_torch_output.shape}"  
  
  
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)  
def test_cab_memory_config(device):  
    """Test different memory configurations"""  
    torch.manual_seed(0)  
  
    num_feat = 64  
    compress_ratio = 3  
    squeeze_factor = 30  
    batch_size = 1  
    height, width = 32, 32  
  
    ref_model = CAB(num_feat=num_feat, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)  
    ref_model.eval()  
  
    input_tensor = torch.randn(batch_size, num_feat, height, width)  
  
    with torch.no_grad():  
        ref_output = ref_model(input_tensor)  
  
    # Test with L1 memory config  
    parameters = preprocess_model_parameters(  
        initialize_model=lambda: ref_model,  
        custom_preprocessor=create_cab_preprocessor(device),  
        device=device,  
    )  
  
    tt_model_l1 = TTCAB(  
        device=device,  
        parameters=parameters,  
        num_feat=num_feat,  
        compress_ratio=compress_ratio,  
        squeeze_factor=squeeze_factor,  
        memory_config=ttnn.L1_MEMORY_CONFIG,  
    )  
  
    tt_input = ttnn.from_torch(  
        input_tensor.permute(0, 2, 3, 1),  
        device=device,  
        layout=ttnn.TILE_LAYOUT,  
        dtype=ttnn.bfloat16  
    )  
  
    tt_output_l1 = tt_model_l1(tt_input)  
    tt_torch_output_l1 = ttnn.to_torch(tt_output_l1).permute(0, 3, 1, 2)  
  
    does_pass_l1, pcc_message_l1 = comp_pcc(ref_output, tt_torch_output_l1, 0.97)  
    logger.info(f"L1 Memory Config: {pcc_message_l1}")  
    assert does_pass_l1, f"L1 memory config failed: {pcc_message_l1}"  
  
  
@pytest.mark.parametrize(  
    "compress_ratio, squeeze_factor",  
    [  
        (2, 16),   # High compression  
        (4, 32),   # Low compression  
        (3, 8),    # High squeeze  
        (6, 64),   # Low squeeze  
    ],  
)  
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)  
def test_cab_compression_ratios(device, compress_ratio, squeeze_factor):  
    """Test different compression and squeeze factor configurations"""  
    torch.manual_seed(0)  
  
    num_feat = 96  
    batch_size = 1  
    height, width = 32, 32  
  
    ref_model = CAB(num_feat=num_feat, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)  
    ref_model.eval()  
  
    input_tensor = torch.randn(batch_size, num_feat, height, width)  
  
    with torch.no_grad():  
        ref_output = ref_model(input_tensor)  
  
    parameters = preprocess_model_parameters(  
        initialize_model=lambda: ref_model,  
        custom_preprocessor=create_cab_preprocessor(device),  
        device=device,  
    )  
  
    tt_model = TTCAB(  
        device=device,  
        parameters=parameters,  
        num_feat=num_feat,  
        compress_ratio=compress_ratio,  
        squeeze_factor=squeeze_factor,  
        memory_config=ttnn.DRAM_MEMORY_CONFIG,  
    )  
  
    tt_input = ttnn.from_torch(  
        input_tensor.permute(0, 2, 3, 1),  
        device=device,  
        layout=ttnn.TILE_LAYOUT,  
        dtype=ttnn.bfloat16  
    )  
  
    tt_output = tt_model(tt_input)  
    tt_torch_output = ttnn.to_torch(tt_output).permute(0, 3, 1, 2)  
  
    does_pass, pcc_message = comp_pcc(ref_output, tt_torch_output, 0.97)  
  
    logger.info(f"Compress ratio: {compress_ratio}, Squeeze factor: {squeeze_factor}")  
    logger.info(f"Reference output shape: {ref_output.shape}")  
    logger.info(f"TTNN output shape: {tt_torch_output.shape}")  
    logger.info(pcc_message)  
  
    if does_pass:  
        logger.info("CAB Compression Test Passed!")  
    else:  
        logger.warning("CAB Compression Test Failed!")  
  
    assert does_pass, f"PCC check failed: {pcc_message}"  
  
  
def test_cab_multiple_iterations(device):  
    """Test multiple forward passes to check for memory leaks"""  
    torch.manual_seed(0)  
  
    num_feat = 128  
    compress_ratio = 4  
    squeeze_factor = 32  
    batch_size = 1  
    height, width = 64, 64  
  
    ref_model = CAB(num_feat=num_feat, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)  
    ref_model.eval()  
  
    parameters = preprocess_model_parameters(  
        initialize_model=lambda: ref_model,  
        custom_preprocessor=create_cab_preprocessor(device),  
        device=device,  
    )  
  
    tt_model = TTCAB(  
        device=device,  
        parameters=parameters,  
        num_feat=num_feat,  
        compress_ratio=compress_ratio,  
        squeeze_factor=squeeze_factor,  
        memory_config=ttnn.DRAM_MEMORY_CONFIG,  
    )  
  
    # Run multiple forward passes to check for memory leaks  
    for i in range(3):  
        input_tensor = torch.randn(batch_size, num_feat, height, width)  
  
        with torch.no_grad():  
            ref_output = ref_model(input_tensor)  
  
        tt_input = ttnn.from_torch(  
            input_tensor.permute(0, 2, 3, 1),  
            device=device,  
            layout=ttnn.TILE_LAYOUT,  
            dtype=ttnn.bfloat16  
        )  
  
        tt_output = tt_model(tt_input)  
        tt_torch_output = ttnn.to_torch(tt_output).permute(0, 3, 1, 2)  
  
        does_pass, pcc_message = comp_pcc(ref_output, tt_torch_output, 0.97)  
        logger.info(f"Iteration {i+1}: {pcc_message}")  
        assert does_pass, f"Iteration {i+1} failed: {pcc_message}"  
  
        # Clean up tensors to prevent memory leaks  
        ttnn.deallocate(tt_input)  
        ttnn.deallocate(tt_output)  
  
        logger.info(f"Iteration {i+1} completed successfully")