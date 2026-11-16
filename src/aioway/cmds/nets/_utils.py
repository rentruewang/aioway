# Copyright (c) AIoWay Authors - All Rights Reserved

import enum

import torch
from torch import nn

from aioway._signs import Param
from aioway.ops import FuncInputOp, LayerOp, Op, ViewOp
from aioway.t import FloatTensorDomain

__all__ = ["observ_op", "linear_tree", "Activation"]


def observ_op() -> FuncInputOp[torch.Tensor]:
    "The placeholder for the observation, the input of the emitted `OpFunc`."

    return FuncInputOp(
        param=Param.build("observ"), typ=torch.Tensor, domain=FloatTensorDomain()
    )


def linear_tree(
    input: Op[torch.Tensor], in_shape: torch.Size, out_shape: torch.Size
) -> Op:
    """
    `Linear` from `in_shape` to `out_shape`, both without the batch dimension.

    Flattens the input and unflattens the output if they are not 1D.
    """

    if len(in_shape) != 1:
        input = ViewOp(nn.Flatten(), input)

    op = LayerOp(nn.Linear(in_shape.numel(), out_shape.numel()), input)

    if len(out_shape) != 1:
        op = ViewOp(nn.Unflatten(-1, out_shape), op)

    return op


class Activation(enum.StrEnum):
    """
    The activation enum.
    """

    RELU = "relu"
    RELU6 = "relu6"
    CELU = "celu"
    GELU = "gelu"
    SIGMOID = "sigmoid"
    TANH = "tanh"

    @property
    def nn_type(self) -> type[nn.Module]:
        "Convert from the `Activation` to `nn.Module` type."

        match self:
            case self.RELU:
                return nn.ReLU
            case self.RELU6:
                return nn.ReLU6
            case self.CELU:
                return nn.CELU
            case self.GELU:
                return nn.GELU
            case self.SIGMOID:
                return nn.Sigmoid
            case self.TANH:
                return nn.Tanh

        raise NotImplementedError(self)
