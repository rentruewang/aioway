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
from .infos import Info, InfoList
from .layouts import Layout, LayoutLike
from .shapes import Shape, ShapeLike

__all__ = ["Attr", "attr", "AttrTensor"]


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

    layout: Layout
    """
    The layout of the tensor.
    """

    infos: InfoList = dcls.field(default_factory=InfoList)
    """
    Extra information about the attribute.
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
            "infos": self.infos,
        }
        return {key: val.__getstate__() for key, val in mapping.items()}

    def __hash__(self) -> int:
        return hash(json.dumps(self.__getstate__(), sort_keys=True))

    @typing.override
    def __repr__(self) -> str:
        return f"[shape={self.shape},dtype={self.dtype},device={self.device},infos={self.infos!r}]"

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
        layout: LayoutLike,
        infos: cabc.Iterable[Info] = (),
    ) -> typing.Self:
        """
        The convenient constructor for `Attr`.

        Args:
            device: Things that can be converted to `Device`.
            dtype: Things that can be converted to `DType`.
            shape: Things that can be converted to `Shape`.
            layout: Things that can be converted to `Layout`.

        Returns:
            An attribute instance.
        """

        return cls(
            device=Device.parse(device),
            dtype=DType.parse(dtype),
            shape=Shape.parse(shape),
            layout=Layout.parse(layout),
            infos=InfoList(*infos),
        )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, /, *infos: Info) -> typing.Self:
        "Parse the `torch.Tensor`'s `Attr` representation"

        return cls.parse(
            device=tensor.device,
            shape=tensor.shape,
            dtype=tensor.dtype,
            layout=tensor.layout,
            infos=infos,
        )


@dcls.dataclass(frozen=True)
class AttrTensor:
    """
    `AttrTensor` stores a `torch.Tensor` and a precomputed `Attr`.
    """

    tensor: torch.Tensor
    """
    The tensor that needs info attached.
    """

    attr: Attr
    """
    The attached info (most of the info can be derived from the tensor, but some cannot).
    """

    @classmethod
    def from_tensor_info(cls, tensor: torch.Tensor, *infos: Info) -> typing.Self:
        """
        Auto determine some fields in `Attr` and create an `AttrTensor`.
        """

        attr = Attr.from_tensor(tensor, *infos)
        return cls(tensor=tensor, attr=attr)


class AttrDict(typing.TypedDict):
    device: DeviceLike
    dtype: DTypeLike
    shape: ShapeLike
    layout: LayoutLike
    infos: typing.NotRequired[cabc.Iterable[Info]]


type AttrLike = Attr | AttrDict | torch.Tensor


def attr(item: AttrLike, /) -> Attr:
    "The convenient constructor function for `Attr` to convert from similar types."

    if isinstance(item, Attr):
        return item

    if isinstance(item, torch.Tensor):
        return Attr.from_tensor(item)

    if _is_attr_dict(item):
        return Attr.parse(
            device=item["device"],
            shape=item["shape"],
            dtype=item["dtype"],
            layout=item["layout"],
            infos=item.get("infos", ()),
        )

    raise TypeError(
        f"Do not know how to handle {item=}, {type(item)=}, because it is malformed."
    )


@typing.no_type_check
def _is_attr_dict(item: object) -> typing.TypeGuard[AttrDict]:

    if not isinstance(item, cabc.Mapping):
        return False

    try:
        _ = Attr.parse(
            device=item["device"],
            dtype=item["dtype"],
            shape=item["shape"],
            layout=item["layout"],
            infos=item.get("infos", ()),
        )
    except Exception:
        return False

    return True
