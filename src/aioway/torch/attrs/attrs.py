# Copyright (c) AIoWay Authors - All Rights Reserved

"Schema is a collection of metadata describing the 'type' of data."

import dataclasses as dcls
import json
import typing
from collections import abc as cabc

import loguru as L
import tensordict as td
import torch

from aioway._utils import is_tuple_of
from aioway.torch._utils import dcls_asdict

from .devices import Device, DeviceLike
from .dtypes import DType, DTypeLike
from .layouts import Layout, LayoutLike
from .shapes import Shape, ShapeLike

__all__ = ["Attr", "AttrDict", "parse_attr"]


type AttrCompat = Attr | AttrLike | AttrLikeMapping | torch.Tensor
type AttrDictCompat = AttrDict | td.TensorDict | td.TensorClass | cabc.Mapping


@typing.runtime_checkable
class AttrLike(typing.Protocol):
    dtype: DType
    shape: Shape
    device: Device
    requires_grad: bool
    layout: Layout


class AttrLikeMapping(typing.TypedDict):
    shape: ShapeLike
    dtype: DTypeLike
    device: typing.NotRequired[DeviceLike]
    requires_grad: typing.NotRequired[bool]
    layout: typing.NotRequired[LayoutLike]


@dcls.dataclass(frozen=True, eq=False)
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
            parsed = parse_attr(other)
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

    def to_fake(self) -> torch.Tensor:
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


class AttrDict:
    """
    `AttrDict` is a `dict[str, Attr]` with additional utilities.
    """

    def __init__(self, mapping: dict[str, typing.Any] | None = None) -> None:
        self._schemas = mapping or {}

    def __contains__(self, key) -> bool:
        return key in self.keys(include_nested=True)

    def __eq__(self, other) -> bool:
        if isinstance(other, AttrDict):
            return self._schemas == other._schemas

        if isinstance(other, cabc.Mapping):
            return self._schemas == dict(other)

        return NotImplemented

    def __len__(self) -> int:
        return len(self._schemas)

    def __getitem__(self, key: str | tuple[str, ...], /) -> Attr | AttrDict:
        if isinstance(key, str):
            return self._schemas[key]

        try:
            return self._getitem_recurse(*key)
        except KeyError, AssertionError:
            raise KeyError(key)

    def __getstate__(self) -> dict:
        return {key: val.__getstate__() for key, val in self.items()}

    def __hash__(self) -> int:
        return hash(json.dumps(self.__getstate__(), sort_keys=True))

    @typing.overload
    def get[D](self, key: str | tuple[str, ...], default: D) -> Attr | AttrDict | D: ...

    @typing.overload
    def get(self, key: str | tuple[str, ...]) -> Attr | AttrDict | None: ...

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def _getitem_recurse(self, *key: str):
        assert key

        first, *rest = key

        try:
            child = self[first]
        except KeyError:
            raise KeyError

        # Still more to recurse. Must be `Schema`.
        if rest:
            assert isinstance(child, AttrDict)
            return child._getitem_recurse(*rest)

        # If it's tuple of 1 level (only first), we have reached the end.
        else:
            assert isinstance(child, Attr)
            return child

    def keys(self, *, leaves_only: bool = False, include_nested: bool = False):
        if include_nested and leaves_only:
            raise ValueError("`leaves_only` and `include_nested` cannot both be true.")

        # Always yield the non nested keys.
        for key, val in self._items_of_type(Attr):
            yield key

        if leaves_only:
            return

        # This is the branch where it would yield sub-schema keys as tuples.
        if include_nested:
            yield from self._nested_keys()

        for key, _ in self._items_of_type(AttrDict):
            yield key

    def _nested_keys(self):
        L.logger.trace("Recurse into sub-Schema of {} and yield tuples", self)

        for key, val in self._items_of_type(AttrDict):
            for child_key in val.keys(include_nested=True):
                child_key = child_key if isinstance(child_key, tuple) else (child_key,)
                assert is_tuple_of(str)(child_key)
                yield key, *child_key

    def values(self, *, leaves_only: bool = False, include_nested: bool = False):
        for key in self.keys(leaves_only=leaves_only, include_nested=include_nested):
            yield self[key]

    def items(self, *, leaves_only: bool = False, include_nested: bool = False):
        for key in self.keys(leaves_only=leaves_only, include_nested=include_nested):
            yield key, self[key]

    def _items_of_type[T](self, typ: type[T]) -> cabc.Generator[tuple[str, T]]:
        for key, child in self._schemas.items():
            if isinstance(child, typ):
                yield key, child

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

        return any(child.requires_grad for child in self.values())

    def select(self, *cols: str, strict: bool = False) -> typing.Self:
        """
        Select subset of columns in the `Schema`.
        If `strict`, all keys should be present, or `KeyValue` would be raised.
        """

        result = type(self)({key: val for key, val in self.items() if key in cols})

        if strict and len(result) != len(cols):
            not_found = [col for col in cols if col not in result]
            raise KeyError(
                f"These keys: {not_found} are not found, which is disallowed in strict mode."
            )

        return result

    def to_fake(self) -> td.TensorDict:
        "Convert `Schema` to a fake `td.TensorDict`."

        # Both `Attr` and `Schema` have `to_fake`.
        return td.TensorDict({key: attr.to_fake() for key, attr in self.items()})


def _parse_attr(item: typing.Any, /) -> Attr:
    """
    The convenient constructor function for `Attr` to convert from similar types.

    Returns `NotImplemented` for unhandled objects.
    """

    if isinstance(item, Attr):
        return item

    if isinstance(item, torch.Tensor):
        return _attr_from_tensor(item)

    if isinstance(item, dict) and (attr := _attr_from_dict(item)):
        return attr

    return NotImplemented


def _parse_attr_dict(mapping: AttrDictCompat, /) -> AttrDict:
    if isinstance(mapping, AttrDict):
        return mapping

    if isinstance(mapping, cabc.Mapping) or td.is_tensor_collection(mapping):
        mapping = typing.cast(cabc.Mapping, mapping)
        return AttrDict({key: parse_attr(tensor) for key, tensor in mapping.items()})

    return NotImplemented


@typing.overload
def parse_attr(item: AttrCompat) -> Attr: ...


@typing.overload
def parse_attr(item: AttrDictCompat) -> AttrDict: ...


def parse_attr(item):
    if (attr := _parse_attr(item)) is not NotImplemented:
        return attr

    if isinstance(item, cabc.Mapping) or td.is_tensor_collection(item):
        return _parse_attr_dict(item)

    raise TypeError(type(item))


def _attr_from_dict(item) -> Attr | None:
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


def _attr_from_tensor(tensor: torch.Tensor, /) -> Attr:
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
