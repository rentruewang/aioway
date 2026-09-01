# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway.emits import sample_from_tspec
import torch
from aioway.modes import fake_mode
from aioway.instrs import Instr
import abc
import dataclasses as dcls
import math
import typing
from torchrl.data import tensor_specs as tspecs
from torch import nn
import dataclasses as dcls
from aioway._utils import is_tuple_of

from ..nn import NnInstr, instr_dcls

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


@instr_dcls
class _BaseAvgSliding(NnInstr, abc.ABC):
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


@instr_dcls
class _BaseSliding(_BaseAvgSliding, abc.ABC):
    _: dcls.KW_ONLY

    dilation: int | tuple[int, ...] = 1
    "Spacing between kernel elements. Default: 1."

    def __post_init__(self):
        super().__post_init__()
        _ = _cast_ndim_int(self.NDIM, self.dilation)


@instr_dcls
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


@instr_dcls
class _BaseConv(_BaseSliding, _BaseConvWeights, NnInstr):

    def __post_init__(self) -> None:
        _BaseSliding.__post_init__(self)
        _BaseConvWeights.__post_init__(self)


@instr_dcls
class _BaseAvgPool(_BaseAvgSliding, NnInstr):

    def __post_init__(self) -> None:
        _BaseAvgSliding.__post_init__(self)


@instr_dcls
class _BaseMaxPool(_BaseSliding, NnInstr):

    return_indices: bool = False
    """
    If `True`, will return the max indices along with the outputs.
    Useful for `torch.nn.MaxUnpool*d` later
    """

    def __post_init__(self) -> None:
        _BaseSliding.__post_init__(self)


@instr_dcls
class Conv1d(_BaseConv):
    """
    Applies a 1D convolution over an input signal composed of several input planes.
    """

    NN = nn.Conv1d
    NDIM = 1


@instr_dcls
class Conv2d(_BaseConv):
    """
    Applies a 2D convolution over an input signal composed of several input planes.
    """

    NN = nn.Conv2d
    NDIM = 2


@instr_dcls
class Conv3d(_BaseConv):
    """
    Applies a 3D convolution over an input signal composed of several input planes.
    """

    NN = nn.Conv3d
    NDIM = 3


@instr_dcls
class MaxPool1d(_BaseMaxPool):
    """
    Applies a 1D max pooling over an input signal composed of several input planes.
    """

    NN = nn.MaxPool1d
    NDIM = 1


@instr_dcls
class MaxPool2d(_BaseMaxPool):
    """
    Applies a 2D max pooling over an input signal composed of several input planes.
    """

    NN = nn.MaxPool2d
    NDIM = 2


@instr_dcls
class MaxPool3d(_BaseMaxPool):
    """
    Applies a 3D max pooling over an input signal composed of several input planes.
    """

    NN = nn.MaxPool3d
    NDIM = 3


@instr_dcls
class AvgPool1d(_BaseAvgPool):
    """
    Applies a 1D average pooling over an input signal composed of several input planes.
    """

    NN = nn.AvgPool1d
    NDIM = 1


@instr_dcls
class AvgPool2d(_BaseAvgPool):
    """
    Applies a 2D average pooling over an input signal composed of several input planes.
    """

    NN = nn.AvgPool2d
    NDIM = 2


@instr_dcls
class AvgPool3d(_BaseAvgPool):
    """
    Applies a 3D average pooling over an input signal composed of several input planes.
    """

    NN = nn.AvgPool3d
    NDIM = 3


def _cast_ndim_int(ndim: int, val: int | tuple[int, ...]) -> tuple[int, ...]:
    "Convert int to tuple of ndim. Useful for checking (will raise error if not valid)."

    if isinstance(val, int):
        return (val,) * ndim

    if is_tuple_of(int)(val) and len(val) == ndim:
        return val

    raise ValueError(f"Invalid value: {val}.")


@dcls.dataclass(frozen=True)
class _PoolDeductor:
    allowed_dims: tuple[int, ...]
    "The allowed dimensions."

    # Mark `instr` as `typing.Any` to use with multiple types.
    def __call__(self, instr: typing.Any, input: tspecs.Unbounded) -> tspecs.Unbounded:
        if input.ndim not in self.allowed_dims:
            return NotImplemented

        with fake_mode():
            module = instr.module()
            output: torch.Tensor = module(sample_from_tspec(input))

        batch, *rest = output.shape
        assert batch == input.shape[0]

        return tspecs.Unbounded(torch.Size(rest))

    def register(self, *instrs: type[Instr]) -> None:
        for instr in instrs:
            instr.deductor().register(self)


_PoolDeductor((1, 2)).register(Conv1d, AvgPool1d, MaxPool1d)
_PoolDeductor((3,)).register(Conv2d, AvgPool2d, MaxPool2d)
_PoolDeductor((4,)).register(Conv3d, AvgPool3d, MaxPool3d)
