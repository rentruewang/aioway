# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch import nn

from aioway._utils import Sign
from aioway.cc import nn_sign_skeleton


def _module_type_signs():
    LAYER_SIGN = Sign.from_inputs("input")
    LOSS_SIGN = Sign.from_inputs("input", "target")
    BILINAR_SIGN = Sign.from_inputs("input1", "input2")

    yield nn.AvgPool1d, LAYER_SIGN
    yield nn.AvgPool2d, LAYER_SIGN
    yield nn.AvgPool3d, LAYER_SIGN
    yield nn.BatchNorm1d, LAYER_SIGN
    yield nn.BatchNorm2d, LAYER_SIGN
    yield nn.BatchNorm3d, LAYER_SIGN
    yield nn.CELU, LAYER_SIGN
    yield nn.Conv1d, LAYER_SIGN
    yield nn.Conv2d, LAYER_SIGN
    yield nn.Conv3d, LAYER_SIGN
    yield nn.Dropout1d, LAYER_SIGN
    yield nn.Dropout2d, LAYER_SIGN
    yield nn.Dropout3d, LAYER_SIGN
    yield nn.Embedding, LAYER_SIGN
    yield nn.Flatten, LAYER_SIGN
    yield nn.GELU, LAYER_SIGN
    yield nn.Identity, LAYER_SIGN
    yield nn.InstanceNorm1d, LAYER_SIGN
    yield nn.InstanceNorm2d, LAYER_SIGN
    yield nn.InstanceNorm3d, LAYER_SIGN
    yield nn.LayerNorm, LAYER_SIGN
    yield nn.Linear, LAYER_SIGN
    yield nn.MaxPool1d, LAYER_SIGN
    yield nn.MaxPool2d, LAYER_SIGN
    yield nn.MaxPool3d, LAYER_SIGN
    yield nn.ReLU, LAYER_SIGN
    yield nn.ReLU6, LAYER_SIGN
    yield nn.Sequential, LAYER_SIGN
    yield nn.Sigmoid, LAYER_SIGN
    yield nn.Tanh, LAYER_SIGN
    yield nn.Unflatten, LAYER_SIGN

    yield nn.L1Loss, LOSS_SIGN
    yield nn.SmoothL1Loss, LOSS_SIGN
    yield nn.MSELoss, LOSS_SIGN

    yield nn.Bilinear, BILINAR_SIGN


@pytest.mark.parametrize("module_type,sign", _module_type_signs())
def test_module(module_type: type[nn.Module], sign: Sign):
    assert nn_sign_skeleton(module_type) == sign
