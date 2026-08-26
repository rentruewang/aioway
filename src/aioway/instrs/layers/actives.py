# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn

from ..nn import NnInstr, nn_instr_dcls

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


@nn_instr_dcls
class ReLU(NnInstr):
    "Applies the rectified linear unit function element-wise."

    NN = nn.ReLU


@nn_instr_dcls
class ReLU6(NnInstr):
    "Applies the ReLU6 function element-wise."

    NN = nn.ReLU6


@nn_instr_dcls
class CELU(NnInstr):
    "Applies the CELU function element-wise."

    NN = nn.CELU


@nn_instr_dcls
class GELU(NnInstr):
    "Applies the GELU function element-wise."

    NN = nn.GELU


@nn_instr_dcls
class Sigmoid(NnInstr):
    "Applies the Sigmoid function element-wise."

    NN = nn.Sigmoid


@nn_instr_dcls
class Tanh(NnInstr):
    "Applies the Tanh function element-wise."

    NN = nn.Tanh


@nn_instr_dcls
class Softmin(NnInstr):
    "Applies the Softmin function to an n-dimensional input Tensor."

    NN = nn.Softmin


@nn_instr_dcls
class Softmax(NnInstr):
    "Applies the Softmax function to an n-dimensional input Tensor."

    NN = nn.Softmax


@nn_instr_dcls
class LogSoftmax(NnInstr):
    "Applies the LogSoftmax function to an n-dimensional input Tensor."

    NN = nn.LogSoftmax
