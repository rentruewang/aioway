# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn

from .might import Might

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


class ReLU(Might):
    "Applies the rectified linear unit function element-wise."

    KEY = nn.ReLU


class ReLU6(Might):
    "Applies the ReLU6 function element-wise."

    KEY = nn.ReLU6


class CELU(Might):
    "Applies the CELU function element-wise."

    KEY = nn.CELU


class GELU(Might):
    "Applies the GELU function element-wise."

    KEY = nn.GELU


class Sigmoid(Might):
    "Applies the Sigmoid function element-wise."

    KEY = nn.Sigmoid


class Tanh(Might):
    "Applies the Tanh function element-wise."

    KEY = nn.Tanh


class Softmin(Might):
    "Applies the Softmin function to an n-dimensional input Tensor."

    KEY = nn.Softmin


class Softmax(Might):
    "Applies the Softmax function to an n-dimensional input Tensor."

    KEY = nn.Softmax


class LogSoftmax(Might):
    "Applies the LogSoftmax function to an n-dimensional input Tensor."

    KEY = nn.LogSoftmax
