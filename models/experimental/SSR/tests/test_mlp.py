import torch
import pytest
import math

import ttnn

from loguru import logger

from models.experimental.SSR.reference.SSR.model.net_blocks import Mlp
from models.experimental.SSR.tt import TTMlp

from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_bias, preprocess_linear_weight
from models.utility_functions import (
    tt2torch_tensor,
    comp_pcc,
)


def create_mlp_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if hasattr(torch_model, "fc1") and hasattr(torch_model, "fc2"):  # MLP model
            parameters["fc1"] = {}
            parameters["fc2"] = {}

            # Preprocess fc1 layer parameters
            parameters["fc1"]["weight"] = preprocess_linear_weight(torch_model.fc1.weight, dtype=ttnn.bfloat16)
            parameters["fc1"]["bias"] = preprocess_linear_bias(torch_model.fc1.bias, dtype=ttnn.bfloat16)

            # Preprocess fc2 layer parameters
            parameters["fc2"]["weight"] = preprocess_linear_weight(torch_model.fc2.weight, dtype=ttnn.bfloat16)
            parameters["fc2"]["bias"] = preprocess_linear_bias(torch_model.fc2.bias, dtype=ttnn.bfloat16)

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("image_size, patch_size, token_size, input_shape", ((256, 2, 4, (1, 16, 3072)),))
def test_mlp(image_size, patch_size, token_size, input_shape):
    x = torch.randn(input_shape)

    num_layers = int(math.log2((image_size // patch_size) // token_size))
    ref_layer = Mlp(
        in_features=96 * (2**num_layers),
        hidden_features=96,
        out_features=96,
    )

    ref_output = ref_layer(x)

    device = ttnn.open_device(device_id=0)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_layer, custom_preprocessor=create_mlp_preprocessor(device), device=device
    )

    tt_layer = TTMlp(
        device, None, in_features=96 * (2**num_layers), hidden_features=96, out_features=96, parameters=parameters
    )
    tt_input = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    tt_output = tt_layer(tt_input)
    tt_torch_output = tt2torch_tensor(tt_output)

    does_pass, pcc_message = comp_pcc(ref_output, tt_torch_output, 0.99)

    logger.info(pcc_message)

    if does_pass:
        logger.info("SwinLayer Passed!")
    else:
        logger.warning("SwinLayer Failed!")

    ttnn.close_device(device)

    assert does_pass
