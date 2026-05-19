# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn

from .inits import MessInit

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


class ReLU(MessInit):
    "Applies the rectified linear unit function element-wise."

    KEY = nn.ReLU


class ReLU6(MessInit):
    "Applies the ReLU6 function element-wise."

    KEY = nn.ReLU6


class CELU(MessInit):
    "Applies the CELU function element-wise."

    KEY = nn.CELU


class GELU(MessInit):
    "Applies the GELU function element-wise."

    KEY = nn.GELU


class Sigmoid(MessInit):
    "Applies the Sigmoid function element-wise."

    KEY = nn.Sigmoid


class Tanh(MessInit):
    "Applies the Tanh function element-wise."

    KEY = nn.Tanh


class Softmin(MessInit):
    "Applies the Softmin function to an n-dimensional input Tensor."

    KEY = nn.Softmin


class Softmax(MessInit):
    "Applies the Softmax function to an n-dimensional input Tensor."

    KEY = nn.Softmax


class LogSoftmax(MessInit):
    "Applies the LogSoftmax function to an n-dimensional input Tensor."

    KEY = nn.LogSoftmax
