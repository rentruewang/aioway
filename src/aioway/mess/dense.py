# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn

from .fwds import InputFwd
from .mess import Mess, MessInit, mess_init_dcls

__all__ = []


@mess_init_dcls
class IdentityInit(MessInit): ...


_ = Mess(nn_type=nn.Identity, init=IdentityInit, fwd=InputFwd)
"""
A placeholder identity operator that is argument-insensitive.
"""


@mess_init_dcls
class LinearInit(MessInit):

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


@mess_init_dcls
class BilinearInit(MessInit):

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


_ = Mess(nn_type=nn.Linear, init=LinearInit, fwd=InputFwd)
"""
Apply the transformation A @ x + b.
"""

_ = Mess(nn_type=nn.Bilinear, init=BilinearInit, fwd=InputFwd)
"""
Apply the transformation x1 @ A @ x2 + b.
"""
