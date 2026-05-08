# Copyright (c) AIoWay Authors - All Rights Reserved

"Schema is a collection of metadata describing the 'type' of data."

import collections
import dataclasses as dcls
import logging
import typing
from collections import abc as cabc

import torch

from aioway._common import get_tracker
from aioway.schemas.infos import Info

from .devices import Device, DeviceLike
from .dtypes import DType, DTypeLike
from .infos import Info
from .shapes import Shape, ShapeLike

__all__ = ["Attr", "attr", "AttrTensor"]


LOGGER = logging.getLogger(__name__)
TRACKER = get_tracker(lambda: Attr)


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

    infos: dict[type[Info], list[Info]] = dcls.field(default_factory=dict)
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

        if not all(issubclass(info, Info) for info in self.infos):
            raise TypeError(f"Not all info in {self.infos=} is a `Info`.")

        if not all(
            isinstance(info_inst, info_type)
            for (info_type, info_list) in self.infos.items()
            for info_inst in info_list
        ):
            raise TypeError(
                "Invalid info passed in. Infos should be a mapping of type to list of instances. "
                f"But got {self.infos}."
            )

    @typing.override
    def __repr__(self) -> str:
        return f"[shape={self.shape},dtype={self.dtype},device={self.device}]"

    def memory(self):
        return self.dtype.bits * self.shape.numel()

    def to_tensor(self):
        """
        Generate a random tensor.
        This should be used under fake mode.
        """

        return (
            torch.zeros(self.shape.concrete())
            .to(self.device.torch())
            .to(self.dtype.torch())
        )

    @classmethod
    def parse(
        cls,
        device: DeviceLike,
        dtype: DTypeLike,
        shape: ShapeLike,
        infos: cabc.Iterable[Info] = (),
    ) -> typing.Self:
        """
        The convenient constructor for `Attr`.

        Args:
            device: Things that can be converted to `Device`.
            dtype: Things that can be converted to `DType`.
            shape: Things that can be converted to `Shape`.

        Returns:
            An attribute instance.
        """

        return cls(
            device=Device.parse(device),
            dtype=DType.parse(dtype),
            shape=Shape.parse(shape),
            infos=_categorize_info(infos),
        )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, /, *infos: Info) -> typing.Self:
        "Parse the `torch.Tensor`'s `Attr` representation"

        return cls.parse(
            device=tensor.device,
            shape=tensor.shape,
            dtype=tensor.dtype,
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


class AttrDict(typing.TypedDict):
    device: DeviceLike
    dtype: DTypeLike
    shape: ShapeLike
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
            infos=item.get("infos", ()),
        )
    except Exception:
        return False

    return True


def _categorize_info(
    infos: cabc.Iterable[Info],
) -> dict[type[Info], list[Info]]:
    """
    Organize sequence of `Info` by `type(info)`, into a `dict[type, list]`.
    """

    result: dict[type[Info], list[Info]] = collections.defaultdict(list)

    for info in infos:
        result[type(info)].append(info)

    return result
