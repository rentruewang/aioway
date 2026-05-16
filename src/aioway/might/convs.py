# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import math
import typing

from torch import nn

from aioway._common import dcls_frozen_no_repr
from aioway._common.typing.checks import is_tuple_of

from .might import Might


@dcls_frozen_no_repr
class _BaseConv(Might, abc.ABC):
    KEY: typing.ClassVar[type[nn.Module]] = NotImplemented
    NDIM: typing.ClassVar[int]

    in_channels: int
    "Number of channels in the input image."

    out_channels: int
    "Number of channels in the output image."

    kernel_size: int | tuple[int, ...]
    "Size of the convolving kernel."

    stride: int | tuple[int, ...] = 1
    "Stride of the convolution. Default: 1."

    padding: int | tuple[int, ...] = 0
    "Padding added to both sides of the input. Default: 0"

    dilation: int | tuple[int, ...] = 1
    "Spacing between kernel elements. Default: 1."

    groups: int = 1
    "Number of blocked connections from input channels to output channels. Default: 1."

    bias: bool = True
    "If `True`, adds a learnable bias to the output. Default: `True`."

    padding_mode: str = "zeros"
    "'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'."

    def __post_init__(self) -> None:
        if self.in_channels <= 0:
            raise ValueError(f"{self.in_channels=} <= 0.")
        if self.out_channels <= 0:
            raise ValueError(f"{self.out_channels=} <= 0.")
        if self.in_channels % self.groups or self.out_channels % self.groups:
            gcd = math.gcd(self.in_channels, self.out_channels)
            raise ValueError(
                f"{self.groups=} should divide both {self.in_channels=} and {self.out_channels=}. "
                f"Therefore it should divide {gcd=}."
            )

        # Check these `int`s or `tuple`s if they are valid.
        _ = self._int_as_tuple(self.kernel_size)
        _ = self._int_as_tuple(self.stride)
        _ = self._int_as_tuple(self.padding)
        _ = self._int_as_tuple(self.dilation)

        # Padding should be one of these.
        if self.padding_mode not in (
            opts := ["zeros", "reflect", "replicate", "circular"]
        ):
            raise ValueError(f"{self.padding_mode=!r} should be one of {opts}.")

    @classmethod
    def _int_as_tuple(cls, val: int | tuple[int, ...]) -> tuple[int, ...]:
        if isinstance(val, int):
            return (val,) * cls.NDIM

        if is_tuple_of(int)(val) and len(val) == cls.NDIM:
            return val

        raise ValueError(f"Invalid value: {val}.")


@dcls_frozen_no_repr
class Conv1d(_BaseConv):
    """
    Applies a 1D convolution over an input signal composed of several input planes.
    """

    KEY = nn.Conv1d
    NDIM = 1


@dcls_frozen_no_repr
class Conv2d(_BaseConv):
    """
    Applies a 2D convolution over an input signal composed of several input planes.
    """

    KEY = nn.Conv2d
    NDIM = 2


@dcls_frozen_no_repr
class Conv3d(_BaseConv):
    """
    Applies a 3D convolution over an input signal composed of several input planes.
    """

    KEY = nn.Conv3d
    NDIM = 3
