# Copyright (c) AIoWay Authors - All Rights Reserved

"Schema is a collection of metadata describing the 'type' of data."

import dataclasses as dcls
import json
import logging
import typing
from collections import abc as cabc

import torch

from aioway.fake import torch_fake_mode

from .devices import Device, DeviceLike
from .dtypes import DType, DTypeLike
from .layouts import Layout, LayoutLike
from .shapes import Shape, ShapeLike
from .tags import DimTag, HasDimTag

__all__ = ["Attr", "attr"]


LOGGER = logging.getLogger(__name__)


@dcls.dataclass(frozen=True)
class Attr:
    """
    The "type" for a `torch.Tensor`, describing everything we want to know about it.
    """

    device: Device
    """
    The device for the column.
    """

    dtype: DType
    """
    The data type for the column.
    """

    shape: Shape
    """
    The shape of individual items in the column.
    """

    requires_grad: bool = False
    """
    Whether the tensor requires grad.
    """

    layout: Layout = Layout(torch.strided)
    """
    The layout of the tensor.
    """

    tags: DimTag | None = None
    """
    Dimension tags for `torch.Tensor`.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.device, Device):
            raise TypeError(type(self.device))

        if not isinstance(self.dtype, DType):
            raise TypeError(type(self.dtype))

        if not isinstance(self.shape, Shape):
            raise TypeError(type(self.shape))

    @typing.override
    def __getstate__(self):
        mapping = {
            "device": self.device,
            "dtype": self.dtype,
            "shape": self.shape,
            "tags": self.tags,
        }
        return {key: val.__getstate__() for key, val in mapping.items()}

    def __hash__(self) -> int:
        return hash(json.dumps(self.__getstate__(), sort_keys=True))

    @typing.override
    def __repr__(self) -> str:
        display: list[typing.Any] = [self.shape, self.dtype, self.device]

        # The name "require_grad" is too long.
        if self.requires_grad:
            display.append("grad")

        # You pretty much only see and only use `strided`, so omit it if strided.
        if self.layout != torch.strided:
            display.append(self.layout)

        return "[" + ",".join(map(str, display)) + "]"

    def memory(self):
        return self.dtype.itemsize * self.shape.numel()

    def to_fake_tensor(self):
        """
        Generate a random tensor.
        This should be used under fake mode.
        """

        with torch_fake_mode():
            return (
                torch.zeros(self.shape.unwrap())
                .to(self.device.torch())
                .to(self.dtype.torch())
            )

    @classmethod
    def parse(
        cls,
        device: DeviceLike,
        dtype: DTypeLike,
        shape: ShapeLike,
        requires_grad: bool = False,
        layout: LayoutLike = "strided",
        tags: DimTag | None = None,
    ) -> typing.Self:
        """
        The convenient constructor for `Attr`.

        Args:
            device: Things that can be converted to `Device`.
            dtype: Things that can be converted to `DType`.
            shape: Things that can be converted to `Shape`.
            requires_grad: Boolean value. Default to `False`.
            layout: Things that can be converted to `Layout`. Default to "strided".

        Returns:
            An attribute instance.
        """

        return cls(
            device=Device.parse(device),
            dtype=DType.parse(dtype),
            shape=Shape.parse(shape),
            layout=Layout.parse(layout),
            requires_grad=requires_grad,
            tags=tags,
        )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, /) -> typing.Self:
        "Parse the `torch.Tensor`'s `Attr` representation"

        tags = tensor.__aioway_dim_tag__ if isinstance(tensor, HasDimTag) else None

        return cls.parse(
            device=tensor.device,
            shape=tensor.shape,
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
            tags=tags,
        )


class AttrDict(typing.TypedDict):
    device: DeviceLike
    dtype: DTypeLike
    shape: ShapeLike
    requires_grad: typing.NotRequired[bool]
    layout: typing.NotRequired[LayoutLike]
    infos: typing.NotRequired[DimTag]


type AttrLike = Attr | AttrDict | torch.Tensor


def attr(item: AttrLike, /) -> Attr:
    "The convenient constructor function for `Attr` to convert from similar types."

    if isinstance(item, Attr):
        return item

    if isinstance(item, torch.Tensor):
        return Attr.from_tensor(item)

    if (attr := _is_attr_dict(item)) is not None:
        return attr

    raise TypeError(
        f"Do not know how to handle {item=}, {type(item)=}, because it is malformed."
    )


@typing.no_type_check
def _is_attr_dict(item: object) -> Attr | None:

    if not isinstance(item, cabc.Mapping):
        return None

    try:
        attr = Attr.parse(
            device=item["device"],
            dtype=item["dtype"],
            shape=item["shape"],
            layout=item.get("layout", torch.strided),
            requires_grad=item.get("requires_grad", False),
            tags=item.get("tags"),
        )
    except Exception:
        return None
    else:
        return attr
