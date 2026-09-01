# Copyright (c) AIoWay Authors - All Rights Reserved

"The linear layers."

import torch
from torch import nn
from torchrl.data import tensor_specs as tspecs

from ..nn import NnInstr, instr_dcls

__all__ = ["Identity", "Linear", "Bilinear"]


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
def linear(linear: Linear, input: tspecs.Unbounded) -> tspecs.Unbounded:
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
