# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn

from .modules import NnInit_, nn_init_dcls
from .ufuncs import NnLayerUFunc

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
class ReLU(NnInit_):
    "Applies the rectified linear unit function element-wise."

    NN = nn.ReLU
    UFUNC = NnLayerUFunc


@nn_init_dcls
class ReLU6(NnInit_):
    "Applies the ReLU6 function element-wise."

    NN = nn.ReLU6
    UFUNC = NnLayerUFunc


@nn_init_dcls
class CELU(NnInit_):
    "Applies the CELU function element-wise."

    NN = nn.CELU
    UFUNC = NnLayerUFunc


@nn_init_dcls
class GELU(NnInit_):
    "Applies the GELU function element-wise."

    NN = nn.GELU
    UFUNC = NnLayerUFunc


@nn_init_dcls
class Sigmoid(NnInit_):
    "Applies the Sigmoid function element-wise."

    NN = nn.Sigmoid
    UFUNC = NnLayerUFunc


@nn_init_dcls
class Tanh(NnInit_):
    "Applies the Tanh function element-wise."

    NN = nn.Tanh
    UFUNC = NnLayerUFunc


@nn_init_dcls
class Softmin(NnInit_):
    "Applies the Softmin function to an n-dimensional input Tensor."

    NN = nn.Softmin
    UFUNC = NnLayerUFunc


@nn_init_dcls
class Softmax(NnInit_):
    "Applies the Softmax function to an n-dimensional input Tensor."

    NN = nn.Softmax
    UFUNC = NnLayerUFunc


@nn_init_dcls
class LogSoftmax(NnInit_):
    "Applies the LogSoftmax function to an n-dimensional input Tensor."

    NN = nn.LogSoftmax
    UFUNC = NnLayerUFunc
