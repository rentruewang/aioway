# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import math
import typing

from torch import nn

from aioway._types import dcls_no_repr
from aioway._typing import is_tuple_of

from .inits import MessInit

__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
]

_PADDING = frozenset(["zeros", "reflect", "replicate", "circular"])
"Valid padding values."


@dcls_no_repr
class _BaseAvgSliding(MessInit, abc.ABC):
    NDIM: typing.ClassVar[int]

    _: dcls.KW_ONLY

    kernel_size: int | tuple[int, ...]
    "Size of the convolving kernel."

    stride: int | tuple[int, ...] = 1
    "Stride of the convolution. Default: 1."

    padding: int | tuple[int, ...] = 0
    "Padding added to both sides of the input. Default: 0."

    @typing.override
    def _check_data(self) -> None:
        super()._check_data()

        # Check these `int`s or `tuple`s if they are valid.
        _ = _cast_ndim_int(self.NDIM, self.kernel_size)
        _ = _cast_ndim_int(self.NDIM, self.stride)
        _ = _cast_ndim_int(self.NDIM, self.padding)


@dcls_no_repr
class _BaseSliding(_BaseAvgSliding, MessInit, abc.ABC):
    _: dcls.KW_ONLY

    dilation: int | tuple[int, ...] = 1
    "Spacing between kernel elements. Default: 1."

    @typing.override
    def _check_data(self):
        super()._check_data()
        _ = _cast_ndim_int(self.NDIM, self.dilation)


@dcls_no_repr
class _BaseConvWeights(MessInit, abc.ABC):
    _: dcls.KW_ONLY

    in_channels: int
    "Number of channels in the input image."

    out_channels: int
    "Number of channels in the output image."

    groups: int = 1
    "Number of blocked connections from input channels to output channels. Default: 1."

    bias: bool = True
    "If `True`, adds a learnable bias to the output. Default: `True`."

    padding_mode: str = "zeros"
    "'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'."

    def _check_data(self) -> None:
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

        # Padding should be one of these.
        if self.padding_mode not in _PADDING:
            raise ValueError(f"{self.padding_mode=!r} should be one of {_PADDING}.")


@dcls_no_repr
class _BaseConv(_BaseSliding, _BaseConvWeights, MessInit):
    def _check_data(self) -> None:
        _BaseSliding._check_data(self)
        _BaseConvWeights._check_data(self)


@dcls_no_repr
class _BaseAvgPool(_BaseAvgSliding, MessInit):
    def _check_data(self) -> None:
        super()._check_data()


@dcls_no_repr
class _BaseMaxPool(_BaseSliding, MessInit):
    return_indices: bool = False
    """
    If `True`, will return the max indices along with the outputs.
    Useful for `torch.nn.MaxUnpool*d` later
    """

    def _check_data(self) -> None:
        super()._check_data()


@dcls_no_repr
class Conv1d(_BaseConv, key=nn.Conv1d):
    """
    Applies a 1D convolution over an input signal composed of several input planes.
    """

    NDIM = 1


@dcls_no_repr
class Conv2d(_BaseConv, key=nn.Conv2d):
    """
    Applies a 2D convolution over an input signal composed of several input planes.
    """

    NDIM = 2


@dcls_no_repr
class Conv3d(_BaseConv, key=nn.Conv3d):
    """
    Applies a 3D convolution over an input signal composed of several input planes.
    """

    NDIM = 3


@dcls_no_repr
class MaxPool1d(_BaseMaxPool, key=nn.MaxPool1d):
    """
    Applies a 1D max pooling over an input signal composed of several input planes.
    """

    NDIM = 1


@dcls_no_repr
class MaxPool2d(_BaseMaxPool, key=nn.MaxPool2d):
    """
    Applies a 2D max pooling over an input signal composed of several input planes.
    """

    NDIM = 2


@dcls_no_repr
class MaxPool3d(_BaseMaxPool, key=nn.MaxPool3d):
    """
    Applies a 3D max pooling over an input signal composed of several input planes.
    """

    NDIM = 3


@dcls_no_repr
class AvgPool1d(_BaseAvgPool, key=nn.AvgPool1d):
    """
    Applies a 1D average pooling over an input signal composed of several input planes.
    """

    NDIM = 1


@dcls_no_repr
class AvgPool2d(_BaseAvgPool, key=nn.AvgPool2d):
    """
    Applies a 2D average pooling over an input signal composed of several input planes.
    """

    NDIM = 2


@dcls_no_repr
class AvgPool3d(_BaseAvgPool, key=nn.AvgPool3d):
    """
    Applies a 3D average pooling over an input signal composed of several input planes.
    """

    NDIM = 3


def _cast_ndim_int(ndim: int, val: int | tuple[int, ...]) -> tuple[int, ...]:
    "Convert int to tuple of ndim. Useful for checking (will raise error if not valid)."

    if isinstance(val, int):
        return (val,) * ndim

    if is_tuple_of(int)(val) and len(val) == ndim:
        return val

    raise ValueError(f"Invalid value: {val}.")
