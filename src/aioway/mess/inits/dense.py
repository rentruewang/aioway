# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn

from aioway._types import dcls_no_repr

from .inits import MessInit

__all__ = ["Identity", "Linear", "Bilinear"]


@dcls_no_repr
class Identity(MessInit, key=nn.Identity):
    """
    A placeholder identity operator that is argument-insensitive.
    """


@dcls_no_repr
class Linear(MessInit, key=nn.Linear):
    """
    Apply the transformation A @ x + b.
    """

    in_features: int
    "The size of each input sample, must be > 0."

    out_features: int
    "The size of each output sample, must be > 0"

    bias: bool = True
    "Whether to include the bias terms or not."

    @typing.override
    def _check_data(self):
        if self.in_features <= 0:
            raise ValueError(f"{self.in_features=} <= 0.")

        if self.out_features <= 0:
            raise ValueError(f"{self.out_features=} <= 0.")


@dcls_no_repr
class Bilinear(MessInit, key=nn.Bilinear):
    """
    Apply the transformation x1 @ A @ x2 + b.
    """

    in1_features: int
    "The size of each first input sample, must be > 0."

    in2_features: int
    "The size of each second input sample, must be > 0."

    out_features: int
    "The size of each output sample, must be > 0."

    bias: bool = True
    "If set to `False`, the layer will not learn an additive bias. Default: `True`."

    @typing.override
    def _check_data(self):
        if self.in1_features <= 0:
            raise ValueError(f"{self.in1_features=} <= 0.")

        if self.in2_features <= 0:
            raise ValueError(f"{self.in2_features=} <= 0.")

        if self.out_features <= 0:
            raise ValueError(f"{self.out_features=} <= 0.")
