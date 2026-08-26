# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn

from .modules import NnInstr, nn_init_dcls

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
class ReLU(NnInstr):
    "Applies the rectified linear unit function element-wise."

    NN = nn.ReLU


@nn_init_dcls
class ReLU6(NnInstr):
    "Applies the ReLU6 function element-wise."

    NN = nn.ReLU6


@nn_init_dcls
class CELU(NnInstr):
    "Applies the CELU function element-wise."

    NN = nn.CELU


@nn_init_dcls
class GELU(NnInstr):
    "Applies the GELU function element-wise."

    NN = nn.GELU


@nn_init_dcls
class Sigmoid(NnInstr):
    "Applies the Sigmoid function element-wise."

    NN = nn.Sigmoid


@nn_init_dcls
class Tanh(NnInstr):
    "Applies the Tanh function element-wise."

    NN = nn.Tanh


@nn_init_dcls
class Softmin(NnInstr):
    "Applies the Softmin function to an n-dimensional input Tensor."

    NN = nn.Softmin


@nn_init_dcls
class Softmax(NnInstr):
    "Applies the Softmax function to an n-dimensional input Tensor."

    NN = nn.Softmax


@nn_init_dcls
class LogSoftmax(NnInstr):
    "Applies the LogSoftmax function to an n-dimensional input Tensor."

    NN = nn.LogSoftmax
