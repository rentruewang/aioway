# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn

from .inits import NnInit, nn_init_dcls

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


@nn_init_dcls
class ReLU6(NnInit):
    "Applies the ReLU6 function element-wise."

    NN = nn.ReLU6


@nn_init_dcls
class CELU(NnInit):
    "Applies the CELU function element-wise."

    NN = nn.CELU


@nn_init_dcls
class GELU(NnInit):
    "Applies the GELU function element-wise."

    NN = nn.GELU


@nn_init_dcls
class Sigmoid(NnInit):
    "Applies the Sigmoid function element-wise."

    NN = nn.Sigmoid


@nn_init_dcls
class Tanh(NnInit):
    "Applies the Tanh function element-wise."

    NN = nn.Tanh


@nn_init_dcls
class Softmin(NnInit):
    "Applies the Softmin function to an n-dimensional input Tensor."

    NN = nn.Softmin


@nn_init_dcls
class Softmax(NnInit):
    "Applies the Softmax function to an n-dimensional input Tensor."

    NN = nn.Softmax


@nn_init_dcls
class LogSoftmax(NnInit):
    "Applies the LogSoftmax function to an n-dimensional input Tensor."

    NN = nn.LogSoftmax
