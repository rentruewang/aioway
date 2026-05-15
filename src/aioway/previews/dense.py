# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn

from aioway._common import dcls_frozen_no_repr

from .previews import Preview

__all__ = ["Linear", "Bilinear"]


@dcls_frozen_no_repr
class Linear(Preview):
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


@dcls_frozen_no_repr
class Bilinear(Preview):
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
