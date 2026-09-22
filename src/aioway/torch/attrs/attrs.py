# Copyright (c) AIoWay Authors - All Rights Reserved

"Schema is a collection of metadata describing the 'type' of data."

import dataclasses as dcls
import json
import typing
from collections import abc as cabc

import torch

from aioway._utils import dcls_asdict

from .devices import Device, DeviceLike
from .dtypes import DType, DTypeLike
from .layouts import Layout, LayoutLike
from .shapes import Shape, ShapeLike

__all__ = ["Attr", "attr_dcls", "AttrCompat", "attr_from_tensor"]


type AttrCompat = Attr | AttrLike | AttrLikeDict | torch.Tensor


@typing.runtime_checkable
class AttrLike(typing.Protocol):
    dtype: DType
    shape: Shape
    device: Device
    requires_grad: bool
    layout: Layout


class AttrLikeDict(typing.TypedDict):
    shape: ShapeLike
    dtype: DTypeLike
    device: typing.NotRequired[DeviceLike]
    requires_grad: typing.NotRequired[bool]
    layout: typing.NotRequired[LayoutLike]


@typing.dataclass_transform(frozen_default=True, eq_default=False)
def attr_dcls(cls):
    return dcls.dataclass(frozen=True, eq=False)(cls)


@attr_dcls
class Attr:
    """
    The "type" for a `torch.Tensor`, describing everything we want to know about it.

    A normal `torch.Tensor` consists of 5 attributes,
    `device`, `dtype`, `shape`, `layout`, `requires_grad`.

    The `Attr` type allows these `torch` specific types to work with common types.
    """

    _: dcls.KW_ONLY

    dtype: DType
    """
    The data type for the column.
    """

    shape: Shape
    """
    The shape of individual items in the column.
    """

    device: Device = Device.parse("cpu")
    """
    The device for the column.
    """

    requires_grad: bool = False
    """
    Whether the tensor requires grad.
    """

    layout: Layout = Layout.parse(torch.strided)
    """
    The layout of the tensor.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.device, Device):
            raise TypeError(type(self.device))

        if not isinstance(self.dtype, DType):
            raise TypeError(type(self.dtype))

        if not isinstance(self.shape, Shape):
            raise TypeError(type(self.shape))

    def __bool__(self) -> typing.Literal[True]:
        return True

    @typing.override
    def __eq__(self, other: typing.Any, /) -> bool:
        # Convert to dictionary for faster comparison.
        # This is essentially the default comparison of dataclass,
        # but since we need functionality like try parsing, workaround is needed.
        if isinstance(other, Attr):
            return dcls_asdict(self) == dcls_asdict(other)

        try:
            parsed = self.parse(other)
        except TypeError:
            return NotImplemented
        else:
            return self == parsed

    @typing.override
    def __getstate__(self):
        return {
            "dtype": self.dtype.__getstate__(),
            "shape": self.shape.__getstate__(),
            "device": self.device.__getstate__(),
            "requires_grad": self.requires_grad,
            "layout": self.layout.__getstate__(),
        }

    def __hash__(self) -> int:
        return hash(json.dumps(self.__getstate__(), sort_keys=True))

    @typing.override
    def __repr__(self) -> str:
        display: list[typing.Any] = [self.shape, self.dtype, self.device]

        # You pretty much only see and only use `strided`, so omit it if strided.
        if self.layout != torch.strided:
            display.append(self.layout)

        # The name "require_grad" is too long.
        if self.requires_grad:
            display.append("grad")

        return "{" + ",".join(map(str, display)) + "}"

    def memory(self):
        return self.dtype.itemsize * self.shape.numel()

    def to_fake_tensor(self) -> torch.Tensor:
        """
        Generate a random tensor. This should be used under fake mode.
        """

        from aioway.torch import fake_mode

        with fake_mode():
            return torch.zeros(
                self.shape.torch(),
                dtype=self.dtype.torch(),
                device=self.device.torch(),
                layout=self.layout.torch(),
                requires_grad=self.requires_grad,
            )

    def set_dims(self, dims: cabc.Mapping[int, int]) -> typing.Self:
        "Set shape dims according to `dims` dictionary."

        new_shape = self.shape.set_dims(dims)
        return dcls.replace(self, shape=new_shape)

    @property
    def ndim(self) -> int:
        return self.shape.ndim

    @classmethod
    def parse(cls, item: typing.Any, /) -> Attr:
        """
        The convenient constructor function for `Attr` to convert from similar types.

        Returns `NotImplemented` for unhandled objects.
        """

        if isinstance(item, Attr):
            return item

        if isinstance(item, torch.Tensor):
            return attr_from_tensor(item)

        if isinstance(item, dict) and (attr := attr_from_dict(item)):
            return attr

        return NotImplemented


def attr_from_dict(item) -> Attr | None:
    """
    Attempt to parse a mapping into an `Attr`.

    If illegal values are encoutered, return `None` (so can be used in `if` `else`).
    """

    if not isinstance(item, cabc.Mapping):
        return None

    try:
        attr = _build_attr(
            dtype=item["dtype"],
            shape=item["shape"],
            device=item.get("device", "cpu"),
            layout=item.get("layout", torch.strided),
            requires_grad=item.get("requires_grad", False),
        )
    except Exception:
        return None
    else:
        return attr


def attr_from_tensor(tensor: torch.Tensor, /) -> Attr:
    "Parse the `torch.Tensor`'s `Attr` representation"

    return _build_attr(
        device=tensor.device,
        shape=tensor.shape,
        dtype=tensor.dtype,
        layout=tensor.layout,
        requires_grad=tensor.requires_grad,
    )


def _build_attr(
    dtype: DTypeLike,
    shape: ShapeLike,
    device: DeviceLike,
    requires_grad: bool,
    layout: LayoutLike,
) -> Attr:
    """
    The convenient constructor for `Attr`.

    Args:
        dtype: Things that can be converted to `DType`.
        shape: Things that can be converted to `Shape`.
        device: Things that can be converted to `Device`. Default to "cpu".
        requires_grad: Boolean value. Default to `False`.
        layout: Things that can be converted to `Layout`. Default to "strided".

    Returns:
        An attribute instance.
    """

    return Attr(
        device=Device.parse(device),
        dtype=DType.parse(dtype),
        shape=Shape.parse(shape),
        layout=Layout.parse(layout),
        requires_grad=requires_grad,
    )
