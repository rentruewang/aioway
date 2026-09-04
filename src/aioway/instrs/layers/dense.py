# Copyright (c) AIoWay Authors - All Rights Reserved

"The linear layers."

from torch import nn

from ..nn import NnInstr, instr_dcls

__all__ = ["Identity", "Flatten", "Unflatten", "Linear", "Bilinear"]


@instr_dcls
class Identity(NnInstr):
    """
    A placeholder identity operator that is argument-insensitive.
    """

    NN = nn.Identity


@instr_dcls
class Flatten(NnInstr):
    """
    `Flatten` flattens the input tensor. This is a wrapper to `nn.Flatten`.
    """

    NN = nn.Flatten


@instr_dcls
class Unflatten(NnInstr):
    """
    `Unflatten` un-flattens the input tensor.
    """

    NN = nn.Unflatten

    dim: int
    "The dimension."

    unflattened_size: tuple[int, ...]
    "The sizes to unflatten to."


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
