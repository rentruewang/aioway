# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn

from aioway.tspecs import TSpecInfer

from ..nn import NnInstr, instr_dcls

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


@instr_dcls
class _ActivationInstr(NnInstr):
    @typing.override
    def __tspec_infer__(self) -> TSpecInfer:
        """
        Activation does not change the input.
        For relu, it does change from unbounded to bounded,
        but modelling it as unbounded -> unbounded will do now.
        """

        return lambda t: t


@instr_dcls
class ReLU(_ActivationInstr):
    "Applies the rectified linear unit function element-wise."

    NN = nn.ReLU


@instr_dcls
class ReLU6(_ActivationInstr):
    "Applies the ReLU6 function element-wise."

    NN = nn.ReLU6


@instr_dcls
class CELU(_ActivationInstr):
    "Applies the CELU function element-wise."

    NN = nn.CELU


@instr_dcls
class GELU(_ActivationInstr):
    "Applies the GELU function element-wise."

    NN = nn.GELU


@instr_dcls
class Sigmoid(_ActivationInstr):
    "Applies the Sigmoid function element-wise."

    NN = nn.Sigmoid


@instr_dcls
class Tanh(_ActivationInstr):
    "Applies the Tanh function element-wise."

    NN = nn.Tanh


@instr_dcls
class Softmin(_ActivationInstr):
    "Applies the Softmin function to an n-dimensional input Tensor."

    NN = nn.Softmin


@instr_dcls
class Softmax(_ActivationInstr):
    "Applies the Softmax function to an n-dimensional input Tensor."

    NN = nn.Softmax


@instr_dcls
class LogSoftmax(_ActivationInstr):
    "Applies the LogSoftmax function to an n-dimensional input Tensor."

    NN = nn.LogSoftmax
