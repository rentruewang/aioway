# Copyright (c) AIoWay Authors - All Rights Reserved

"The linear layers."

import typing
from collections import abc as cabc

import torch
from torch import nn
from torchrl.data import tensor_specs as tspecs

from ..nn import NnInstr, instr_dcls

__all__ = ["Identity", "Flatten", "Unflatten", "Linear", "Bilinear"]


@instr_dcls
class Identity(NnInstr):
    """
    A placeholder identity operator that is argument-insensitive.
    """

    NN = nn.Identity


@Identity.deductor().register
def identity_deduct(self, input):
    return input


@instr_dcls
class Flatten(NnInstr):
    """
    `Flatten` flattens the input tensor. This is a wrapper to `nn.Flatten`.
    """

    NN = nn.Flatten


@Flatten.deductor().register
def flatten_deduct(self, input: tspecs.Unbounded):
    return input.flatten(0, -1)


@instr_dcls
class Unflatten(NnInstr):
    """
    `Unflatten` un-flattens the input tensor.
    """

    NN = nn.Unflatten

    dim: int
    "The dimension."

    sizes: cabc.Sequence[int]
    "The sizes to unflatten to."


@Unflatten.deductor().register
@typing.no_type_check
def unflatten_deduct(self: Unflatten, input: tspecs.Unbounded):
    return input.unflatten(self.dim, sizes=torch.Size(self.sizes))


@instr_dcls
class Linear(NnInstr):
    """
    Apply the transformation A @ x + b.
    """

    NN = nn.Linear

    in_features: int
    "The size of each input sample, must be > 0."

    out_features: int
    "The size of each output sample, must be > 0"

    bias: bool = True
    "Whether to include the bias terms or not."

    def __post_init__(self):
        if self.in_features <= 0:
            raise ValueError(f"{self.in_features=} <= 0.")

        if self.out_features <= 0:
            raise ValueError(f"{self.out_features=} <= 0.")


@Linear.deductor().register
def linear_deduct(linear: Linear, input: tspecs.Unbounded) -> tspecs.Unbounded:
    assert input.shape[-1] == linear.in_features
    return tspecs.Unbounded(
        shape=torch.Size([*input.shape[:-1], linear.out_features]), dtype=input.dtype
    )


@instr_dcls
class Bilinear(NnInstr):
    """
    Apply the transformation x1 @ A @ x2 + b.
    """

    NN = nn.Bilinear

    in1_features: int
    "The size of each first input sample, must be > 0."

    in2_features: int
    "The size of each second input sample, must be > 0."

    out_features: int
    "The size of each output sample, must be > 0."

    bias: bool = True
    "If set to `False`, the layer will not learn an additive bias. Default: `True`."

    def __post_init__(self):
        if self.in1_features <= 0:
            raise ValueError(f"{self.in1_features=} <= 0.")

        if self.in2_features <= 0:
            raise ValueError(f"{self.in2_features=} <= 0.")

        if self.out_features <= 0:
            raise ValueError(f"{self.out_features=} <= 0.")


@Bilinear.deductor().register
def bilinear_deduct(
    self: Bilinear, input1: tspecs.Unbounded, input2: tspecs.Unbounded
) -> tspecs.Unbounded:
    input1_shape = input1.shape
    input2_shape = input2.shape

    if input1_shape[:-1] != input2_shape[:-1]:
        return NotImplemented

    return tspecs.Unbounded(shape=torch.Size([*input1_shape[:-1], self.out_features]))
