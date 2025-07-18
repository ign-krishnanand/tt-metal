import torch
import pytest
import math

import ttnn

from loguru import logger

from models.experimental.SSR.reference.SSR.model.net_blocks import Mlp
from models.experimental.SSR.tt import TTMlp

from models.utility_functions import (
    tt2torch_tensor,
    comp_pcc,
)


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

    tt_layer = TTMlp(
        device, None, in_features=96 * (2**num_layers), hidden_features=96, out_features=96, ref_layer=ref_layer
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

    assert does_pass
