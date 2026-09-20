# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch import nn

from aioway._utils import Sign
from aioway.cc import nn_sign_skeleton


def _module_type_signs():
    LAYER_SIGN = Sign.from_inputs("input")
    LOSS_SIGN = Sign.from_inputs("input", "target")
    BILINAR_SIGN = Sign.from_inputs("input1", "input2")

    for layer in _layers():
        yield layer, LAYER_SIGN

    for loss in _losses():
        yield loss, LOSS_SIGN

    yield nn.Bilinear, BILINAR_SIGN


def _losses():
    yield nn.L1Loss()
    yield nn.SmoothL1Loss()
    yield nn.MSELoss()
    yield nn.CrossEntropyLoss()
    yield nn.BCEWithLogitsLoss()
    yield nn.BCELoss()


def _layers():
    yield nn.AvgPool1d
    yield nn.AvgPool2d
    yield nn.AvgPool3d
    yield nn.BatchNorm1d
    yield nn.BatchNorm2d
    yield nn.BatchNorm3d
    yield nn.CELU
    yield nn.Conv1d
    yield nn.Conv2d
    yield nn.Conv3d
    yield nn.Dropout1d
    yield nn.Dropout2d
    yield nn.Dropout3d
    yield nn.Embedding
    yield nn.Flatten
    yield nn.GELU
    yield nn.Identity
    yield nn.InstanceNorm1d
    yield nn.InstanceNorm2d
    yield nn.InstanceNorm3d
    yield nn.LayerNorm
    yield nn.Linear
    yield nn.MaxPool1d
    yield nn.MaxPool2d
    yield nn.MaxPool3d
    yield nn.ReLU
    yield nn.ReLU6
    yield nn.Sequential
    yield nn.Sigmoid
    yield nn.Tanh
    yield nn.Unflatten


@pytest.mark.parametrize("module_type,sign", _module_type_signs())
def test_module(module_type: type[nn.Module], sign: Sign):
    assert nn_sign_skeleton(module_type) == sign
