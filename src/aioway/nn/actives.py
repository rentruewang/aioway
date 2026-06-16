# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn

from aioway.hop import NnLayerHop

from .modules import NnInit, nn_init_dcls

__all__ = [
    "ReLU",
    "ReLU6",
    "CELU",
    "GELU",
    "Sigmoid",
    "Tanh",
    "Softmin",
    "Softmax",
    "LogSoftmax",
]


@nn_init_dcls
class ReLU(NnInit):
    "Applies the rectified linear unit function element-wise."

    NN = nn.ReLU
    HOP = NnLayerHop


@nn_init_dcls
class ReLU6(NnInit):
    "Applies the ReLU6 function element-wise."

    NN = nn.ReLU6
    HOP = NnLayerHop


@nn_init_dcls
class CELU(NnInit):
    "Applies the CELU function element-wise."

    NN = nn.CELU
    HOP = NnLayerHop


@nn_init_dcls
class GELU(NnInit):
    "Applies the GELU function element-wise."

    NN = nn.GELU
    HOP = NnLayerHop


@nn_init_dcls
class Sigmoid(NnInit):
    "Applies the Sigmoid function element-wise."

    NN = nn.Sigmoid
    HOP = NnLayerHop


@nn_init_dcls
class Tanh(NnInit):
    "Applies the Tanh function element-wise."

    NN = nn.Tanh
    HOP = NnLayerHop


@nn_init_dcls
class Softmin(NnInit):
    "Applies the Softmin function to an n-dimensional input Tensor."

    NN = nn.Softmin
    HOP = NnLayerHop


@nn_init_dcls
class Softmax(NnInit):
    "Applies the Softmax function to an n-dimensional input Tensor."

    NN = nn.Softmax
    HOP = NnLayerHop


@nn_init_dcls
class LogSoftmax(NnInit):
    "Applies the LogSoftmax function to an n-dimensional input Tensor."

    NN = nn.LogSoftmax
    HOP = NnLayerHop
