# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import math
import typing

from torch import nn

from aioway._typing import is_tuple_of

from .fwds import InputFwd
from .mess import Mess, MessInit, mess_init_dcls

__all__ = []

_PADDING = frozenset(["zeros", "reflect", "replicate", "circular"])
"Valid padding values."


@mess_init_dcls
class _BaseAvgSliding(abc.ABC):
    KEY: typing.ClassVar[type[nn.Module]] = NotImplemented
    NDIM: typing.ClassVar[int]

    _: dcls.KW_ONLY

    kernel_size: int | tuple[int, ...]
    "Size of the convolving kernel."

    stride: int | tuple[int, ...] = 1
    "Stride of the convolution. Default: 1."

    padding: int | tuple[int, ...] = 0
    "Padding added to both sides of the input. Default: 0."

    def __post_init__(self) -> None:
        # Check these `int`s or `tuple`s if they are valid.
        _ = _cast_ndim_int(self.NDIM, self.kernel_size)
        _ = _cast_ndim_int(self.NDIM, self.stride)
        _ = _cast_ndim_int(self.NDIM, self.padding)


@mess_init_dcls
class _BaseSliding(_BaseAvgSliding, abc.ABC):
    _: dcls.KW_ONLY

    dilation: int | tuple[int, ...] = 1
    "Spacing between kernel elements. Default: 1."

    def __post_init__(self):
        super().__post_init__()
        _ = _cast_ndim_int(self.NDIM, self.dilation)


@mess_init_dcls
class _BaseConvWeights(abc.ABC):
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

        # Padding should be one of these.
        if self.padding_mode not in _PADDING:
            raise ValueError(f"{self.padding_mode=!r} should be one of {_PADDING}.")


def _conv_init(ndim: int):
    @mess_init_dcls
    class ConvInit(_BaseSliding, _BaseConvWeights, MessInit):
        NDIM = ndim

        def __post_init__(self) -> None:
            _BaseSliding.__post_init__(self)
            _BaseConvWeights.__post_init__(self)

    return ConvInit


def _avg_pool_init(ndim: int):
    @mess_init_dcls
    class AvgPoolInit(_BaseAvgSliding, MessInit):
        NIMD = ndim

        def __post_init__(self) -> None:
            _BaseAvgSliding.__post_init__(self)

    return AvgPoolInit


def _max_pool_init(ndim: int):
    @mess_init_dcls
    class MaxPoolInit(_BaseSliding, MessInit):
        NDIM = ndim
        return_indices: bool = False
        """
        If `True`, will return the max indices along with the outputs.
        Useful for `torch.nn.MaxUnpool*d` later
        """

        def __post_init__(self) -> None:
            _BaseSliding.__post_init__(self)

    return MaxPoolInit


_ = Mess(nn_type=nn.Conv1d, init=_conv_init(1), fwd=InputFwd)
"""
Applies a 1D convolution over an input signal composed of several input planes.
"""


_ = Mess(nn_type=nn.Conv2d, init=_conv_init(2), fwd=InputFwd)
"""
Applies a 2D convolution over an input signal composed of several input planes.
"""


_ = Mess(nn_type=nn.Conv3d, init=_conv_init(3), fwd=InputFwd)
"""
Applies a 3D convolution over an input signal composed of several input planes.
"""

_ = Mess(nn_type=nn.MaxPool1d, init=_max_pool_init(1), fwd=InputFwd)
"""
Applies a 1D max pooling over an input signal composed of several input planes.
"""


_ = Mess(nn_type=nn.MaxPool2d, init=_max_pool_init(2), fwd=InputFwd)
"""
Applies a 2D max pooling over an input signal composed of several input planes.
"""


_ = Mess(nn_type=nn.MaxPool3d, init=_max_pool_init(3), fwd=InputFwd)
"""
Applies a 3D max pooling over an input signal composed of several input planes.
"""


_ = Mess(nn_type=nn.AvgPool1d, init=_avg_pool_init(1), fwd=InputFwd)
"""
Applies a 1D average pooling over an input signal composed of several input planes.
"""


_ = Mess(nn_type=nn.AvgPool2d, init=_avg_pool_init(2), fwd=InputFwd)
"""
Applies a 2D average pooling over an input signal composed of several input planes.
"""


_ = Mess(nn_type=nn.AvgPool3d, init=_avg_pool_init(3), fwd=InputFwd)
"""
Applies a 3D average pooling over an input signal composed of several input planes.
"""


def _cast_ndim_int(ndim: int, val: int | tuple[int, ...]) -> tuple[int, ...]:
    "Convert int to tuple of ndim. Useful for checking (will raise error if not valid)."

    if isinstance(val, int):
        return (val,) * ndim

    if is_tuple_of(int)(val) and len(val) == ndim:
        return val

    raise ValueError(f"Invalid value: {val}.")
