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


class ReLU(MessInit, key=nn.ReLU):
    "Applies the rectified linear unit function element-wise."


class ReLU6(MessInit, key=nn.ReLU6):
    "Applies the ReLU6 function element-wise."


class CELU(MessInit, key=nn.CELU):
    "Applies the CELU function element-wise."


class GELU(MessInit, key=nn.GELU):
    "Applies the GELU function element-wise."


class Sigmoid(MessInit, key=nn.Sigmoid):
    "Applies the Sigmoid function element-wise."


class Tanh(MessInit, key=nn.Tanh):
    "Applies the Tanh function element-wise."


class Softmin(MessInit, key=nn.Softmin):
    "Applies the Softmin function to an n-dimensional input Tensor."


class Softmax(MessInit, key=nn.Softmax):
    "Applies the Softmax function to an n-dimensional input Tensor."


class LogSoftmax(MessInit, key=nn.LogSoftmax):
    "Applies the LogSoftmax function to an n-dimensional input Tensor."
