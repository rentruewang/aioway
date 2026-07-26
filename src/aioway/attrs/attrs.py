# Copyright (c) AIoWay Authors - All Rights Reserved

"Schema is a collection of metadata describing the 'type' of data."

import collections
import dataclasses as dcls
import json
import logging
import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway._utils import torch_fake_mode

from .devices import Device, DeviceLike
from .dtypes import DType, DTypeLike
from .layouts import Layout, LayoutLike
from .shapes import Shape, ShapeLike

__all__ = ["Attr", "AttrLike", "AttrDict"]


LOGGER = logging.getLogger(__name__)


type AttrLike = Attr | AttrLikeDict | torch.Tensor


@dcls.dataclass(frozen=True, eq=False)
class Attr:
    """
    The "type" for a `torch.Tensor`, describing everything we want to know about it.

    A normal `torch.Tensor` consists of 5 attributes,
    `device`, `dtype`, `shape`, `layout`, `requires_grad`.

    The `Attr` type allows these `torch` specific types to work with common types.s
    """

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

    def __bool__(self):
        return True

    @typing.override
    @typing.no_type_check
    def __eq__(self, other: object, /) -> bool:
        if isinstance(other, Attr):
            for func in [
                lambda x: x.dtype,
                lambda x: x.shape,
                lambda x: x.device,
                lambda x: x.requires_grad,
                lambda x: x.layout,
            ]:
                if not func(self) == func(other):
                    return False
            else:
                return True

        if parsed := self._try_parse_attr_like_dict(other):
            return self == parsed

        return NotImplemented

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

        with torch_fake_mode():
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
    def build(
        cls,
        dtype: DTypeLike,
        shape: ShapeLike,
        device: DeviceLike = "cpu",
        requires_grad: bool = False,
        layout: LayoutLike = "strided",
    ) -> typing.Self:
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

        return cls(
            device=Device.parse(device),
            dtype=DType.parse(dtype),
            shape=Shape.parse(shape),
            layout=Layout.parse(layout),
            requires_grad=requires_grad,
        )

    @classmethod
    def parse(cls, item: AttrLike, /) -> Attr:
        "The convenient constructor function for `Attr` to convert from similar types."

        if isinstance(item, Attr):
            return item

        if isinstance(item, torch.Tensor):
            return cls.from_tensor(item)

        if (attr := cls._try_parse_attr_like_dict(item)) is not None:
            return attr

        raise TypeError(
            f"Do not know how to handle {item=}, {type(item)=}, because it is malformed."
        )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, /) -> typing.Self:
        "Parse the `torch.Tensor`'s `Attr` representation"

        return cls.build(
            device=tensor.device,
            shape=tensor.shape,
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
        )

    @classmethod
    def _try_parse_attr_like_dict(cls, item: AttrLikeDict) -> typing.Self | None:

        if not isinstance(item, cabc.Mapping):
            return None

        try:
            attr = cls.build(
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


class AttrLikeDict(typing.TypedDict):
    shape: ShapeLike
    dtype: DTypeLike
    device: typing.NotRequired[DeviceLike]
    requires_grad: typing.NotRequired[bool]
    layout: typing.NotRequired[LayoutLike]


class AttrDict(collections.UserDict[str, Attr]):
    """
    `AttrDict` is a `dict[str, Attr]` with additional utilities.
    """

    def __hash__(self):
        return hash(json.dumps({key: val.__getstate__() for key, val in self.items()}))

    @property
    def dtype(self) -> DType | None:
        """
        Get the dtype of the attributes.
        Like `td.TensorDict.dtype`, this is `None` when the types are not homogenious.
        """

        if len(dt := {attr.dtype for attr in self.values()}) != 1:
            return None

        # Get the only one.
        return next(iter(dt))

    @property
    def requires_grad(self) -> bool:
        """
        The `requires_grad`-ness of the `td.TensorDict`.
        It's `True` if any of the attributes is `True`.
        """

        return any(attr.requires_grad for attr in self.values())

    def rename(self, **renames: str) -> typing.Self:
        """
        Renames the current `AttrDict`.
        """

        return type(self)({renames.get(key, key): val for key, val in self.items()})

    def select(self, *cols: str, strict: bool = False) -> typing.Self:
        """
        Select subset of columns in the `AttrDict`.
        If `strict`, all keys should be present, or `KeyValue` would be raised.
        """

        result = type(self)({key: val for key, val in self.items() if key in cols})

        if strict and len(result) != len(cols):
            not_found = [col for col in cols if col not in result]
            raise KeyError(
                f"These keys: {not_found} are not found, which is disallowed in strict mode."
            )

        return result

    def to_fake_tensordict(self) -> td.TensorDict:
        with torch_fake_mode():
            return td.TensorDict(
                {key: attr.to_fake_tensor() for key, attr in self.items()}
            )

    @classmethod
    def parse(cls, mapping: cabc.Mapping[str, AttrLike], /) -> typing.Self:
        return cls({key: Attr.parse(tensor) for key, tensor in mapping.items()})


def attr_dict(mapping: cabc.Mapping[str, AttrLike], /) -> AttrDict:
    return AttrDict.parse(mapping)
