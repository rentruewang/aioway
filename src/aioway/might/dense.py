# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn

from aioway._common import dcls_frozen_no_repr

from .mint import Might

__all__ = ["Linear", "Bilinear"]


@dcls_frozen_no_repr
class Linear(Might):
    """
    Apply the transformation A @ x + b.
    """

    KEY = nn.Linear

    in_features: int
    "The size of each input sample, must be > 0."

    out_features: int
    "The size of each output sample, must be > 0"

    bias: bool = True
    "Whether to include the bias terms or not."


@dcls_frozen_no_repr
class Bilinear(Might):
    """
    Apply the transformation x1 @ A @ x2 + b.
    """

    KEY = nn.Bilinear

    in1_features: int
    "The size of each first input sample, must be > 0."

    in2_features: int
    "The size of each second input sample, must be > 0."

    out_features: int
    "The size of each output sample, must be > 0."

    bias: bool = True
    "If set to `False`, the layer will not learn an additive bias. Default: `True`."
