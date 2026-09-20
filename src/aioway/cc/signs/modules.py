# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn

from .signs import register_nn_type_signature

MODULE_TYPES = (
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.Bilinear,
    nn.CELU,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.Embedding,
    nn.Flatten,
    nn.GELU,
    nn.Identity,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.L1Loss,
    nn.LayerNorm,
    nn.Linear,
    nn.MSELoss,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.ReLU,
    nn.ReLU6,
    nn.Sequential,
    nn.Sigmoid,
    nn.SmoothL1Loss,
    nn.Tanh,
    nn.Unflatten,
)

for _module_type in MODULE_TYPES:
    register_nn_type_signature(_module_type)
