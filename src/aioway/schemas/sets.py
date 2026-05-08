# Copyright (c) AIoWay Authors - All Rights Reserved

"Schema is a collection of metadata describing the 'type' of data."

import dataclasses as dcls
import functools
import logging
import typing
from collections import abc as cabc

import numpy as np
import tensordict as td
import torch

from aioway._common import is_dict_of_str_to, is_list_of

from .attrs import Attr, attr
from .devices import Device, DeviceLike
from .dtypes import DType, DTypeLike
from .shapes import Shape, ShapeLike

__all__ = ["AttrSet", "DTypeSet", "DeviceSet", "ShapeSet", "AttrSetLike", "attr_set"]


LOGGER = logging.getLogger(__name__)

type AttrSetLike = AttrSet | td.TensorDict | dict[str, Attr]


class _AttrItem[T](typing.NamedTuple):
    """
    The name and attribute for each column.
    """

    name: str
    "The name of the column."

    attr: T
    "The attribute that the column has."


@dcls.dataclass(frozen=True, repr=False)
class _AttrSetBase[T](cabc.Mapping[str, T]):
    """
    A set of attributes. Representing the schema in a `td.TensorDict`.

    Right now the columns are in sorted order, but this is not guarenteed.
    Most likely will change in the future.
    """

    attr_items: tuple[_AttrItem[T], ...] = ()
    """
    The data backing the `AttrSet`. Must be sorted.
    """

    def __post_init__(self) -> None:
        if len(self.names) > 1 and not np.all(self.names[:-1] <= self.names[1:]):
            raise ValueError(f"Names are not sorted: {self.names}.")

    @typing.override
    def __repr__(self) -> str:
        return self._repr_string

    def __or__(self, other: cabc.Mapping[str, T]) -> typing.Self:
        return type(self).from_dict({**self, **other})

    @typing.override
    def __len__(self) -> int:
        return len(self.attr_items)

    @typing.override
    def __iter__(self) -> cabc.Iterator[str]:
        return (attr.name for attr in self.attr_items)

    @typing.overload
    def __getitem__(self, key: str) -> T: ...
    @typing.overload
    def __getitem__(self, key: list[str]) -> typing.Self: ...

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.column(key)

        if is_list_of(str)(key):
            return self.select(*key)

        raise KeyError

    @typing.override
    def keys(self):
        return self._keys_view

    def column(self, key: str, /) -> T:
        # Using the `find` function from `AttrSetKeysView`, to be DRY.
        if (idx := self.keys().find(key)) is None:
            raise KeyError(key)

        assert 0 <= idx < len(self)
        return self.attr_items[idx].attr

    def select(self, *keys: str) -> typing.Self:
        return type(self).from_dict({key: self[key] for key in keys})

    @functools.cached_property
    def _repr_string(self):
        kvs = (f"{k}:{v}" for k, v in self.attr_items)
        return "{" + ", ".join(kvs) + "}"

    @functools.cached_property
    def _keys_view(self):
        return _AttrKeysView(self)

    @functools.cached_property
    def names(self):
        return [col.name for col in self.attr_items]

    @classmethod
    def from_values(cls, **mapping: T) -> typing.Self:
        return cls.from_dict(mapping)

    @classmethod
    def from_dict(cls, mapping: cabc.Mapping[str, T], /) -> typing.Self:
        return cls(
            tuple(
                sorted(
                    _AttrItem(name=name, attr=attr) for name, attr in mapping.items()
                )
            )
        )


class DTypeSet(_AttrSetBase[DType]):
    @classmethod
    @typing.override
    def from_dict(cls, mapping: cabc.Mapping[str, DTypeLike], /) -> typing.Self:
        converted = {key: DType.parse(dt) for key, dt in mapping.items()}
        return super().from_dict(converted)


class DeviceSet(_AttrSetBase[Device]):
    @classmethod
    @typing.override
    def from_dict(cls, mapping: cabc.Mapping[str, DeviceLike]) -> typing.Self:
        converted = {key: Device.parse(dev) for key, dev in mapping.items()}
        return super().from_dict(converted)


class ShapeSet(_AttrSetBase[Shape]):
    @classmethod
    @typing.override
    def from_dict(cls, mapping: cabc.Mapping[str, ShapeLike]) -> typing.Self:
        converted = {key: Shape.parse(dev) for key, dev in mapping.items()}
        return super().from_dict(converted)


class AttrSet(_AttrSetBase[Attr]):
    """
    The collection of `Attr`s. This is the data type for a `td.TensorDict`.
    """

    @typing.overload
    def __getitem__(self, idx: str) -> Attr: ...

    @typing.overload
    def __getitem__(
        self, idx: int | slice | list[int] | list[str] | np.ndarray | torch.Tensor
    ) -> typing.Self: ...

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.__getitem_str(idx)

        if is_list_of(str)(idx):
            return self.__getitem_list_str(idx)

        return self.__getitem_batch(idx)

    def __getitem_str(self, idx: str):
        return super().__getitem__(idx)

    def __getitem_list_str(self, idx: list[str]):
        return super().__getitem__(idx)

    def __getitem_batch(self, idx):
        return self.__getitem_batch_impl(idx)

    def __getitem_batch_impl(self, idx):

        if isinstance(idx, str) or is_list_of(str)(idx):
            return super().__getitem__(idx)

        names = self.names
        device_list = self.device_list
        shape_list = self.shape_list
        dtype_list = self.dtype_list

        if isinstance(idx, int):
            new_shape = [shape[1:] for shape in shape_list]

        elif isinstance(idx, slice | np.ndarray | torch.Tensor):
            new_shape = shape_list[:]

        elif isinstance(idx, list) and all(isinstance(i, int) for i in idx):
            new_shape = shape_list[:]

        else:
            raise TypeError(type(idx))

        return self.from_fields(
            names=names,
            device_list=device_list,
            dtype_list=dtype_list,
            shape_list=new_shape,
        )

    def rename(self, **renames: str):
        return self.from_dict(
            {renames.get(key, key): attr for key, attr in self.items()}
        )

    @functools.cached_property
    def dtype_list(self):
        return [col.attr.dtype for col in self.attr_items]

    @functools.cached_property
    def shape_list(self):
        return [col.attr.shape for col in self.attr_items]

    @functools.cached_property
    def device_list(self):
        return [col.attr.device for col in self.attr_items]

    @classmethod
    @typing.no_type_check
    def from_fields(
        cls,
        *,
        names: cabc.Sequence[str],
        shape_list: cabc.Sequence[ShapeLike],
        dtype_list: cabc.Sequence[DTypeLike],
        device_list: cabc.Sequence[DeviceLike],
    ) -> typing.Self:
        "Create an `AttrSet` from a set of seuqences of attributes of same length."

        if not (len(names) == len(shape_list) == len(dtype_list) == len(device_list)):
            raise ValueError(
                "Should all have the same length. "
                f"Got {len(names)=}, {len(shape_list)=}, {len(dtype_list)=}, {len(device_list)=}."
            )

        mapping = {
            name: attr({"device": device, "dtype": dtype, "shape": shape})
            for name, device, dtype, shape in zip(
                names, device_list, dtype_list, shape_list
            )
        }

        return cls.from_dict(mapping)

    @classmethod
    def from_sets(
        cls, *, shape_set: ShapeSet, dtype_set: DTypeSet, device_set: DeviceSet
    ) -> typing.Self:
        "Create an `AttrSet` from `*Set` types. Keys should match."

        shape_keys = shape_set.keys()
        dtype_keys = dtype_set.keys()
        device_keys = device_set.keys()

        if not (shape_keys == dtype_keys == device_keys):
            raise ValueError(
                "All sets should have the same keys. "
                f"Got shapes={shape_keys}, devices={device_keys}, dtypes={dtype_keys}"
            )

        names_list = list(shape_keys)
        dtype_list = [dtype_set[key] for key in names_list]
        device_list = [device_set[key] for key in names_list]
        shape_list = [shape_set.__getitem__(key) for key in names_list]

        return cls.from_fields(
            names=names_list,
            dtype_list=dtype_list,
            device_list=device_list,
            shape_list=shape_list,
        )

    @classmethod
    def from_tensordict(cls, data: td.TensorDict, /) -> typing.Self:
        return cls.from_dict({key: attr(val) for key, val in data.items()})


@dcls.dataclass(frozen=True)
class _AttrKeysView(cabc.KeysView[str]):
    attrset: _AttrSetBase
    "The data to view. Its names must be sorted."

    @typing.override
    def __len__(self):
        return len(self.keys)

    @typing.override
    def __iter__(self):
        return iter(self.keys)

    @typing.override
    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.find(key) is not None

    @typing.override
    @typing.no_type_check
    def __eq__(self, rhs: object) -> bool:
        try:
            rhs_set = set(rhs)

        # If rhs does not have `__iter__`, `TypeError` would be raised.
        except TypeError:
            return False

        return self._set == rhs_set

    @functools.cached_property
    def _set(self):
        return set(self)

    def find(self, key: str) -> int | None:
        """
        Search the `key` in the keys.
        If found, return the index. If not found, return None.
        """

        idx = int(np.searchsorted(self.keys, key))

        if idx < len(self) and self.keys[idx] == key:
            return idx

        return None

    @property
    def keys(self) -> list[str]:
        return self.attrset.names


def attr_set(schema: AttrSetLike) -> AttrSet:
    """
    The convenient constructor for `AttrSet`.
    """

    if isinstance(schema, AttrSet):
        return schema

    if isinstance(schema, td.TensorDict):
        return AttrSet.from_tensordict(schema)

    if is_dict_of_str_to(Attr)(schema):
        return AttrSet.from_dict(schema)

    raise TypeError(
        f"We do not handle the {schema=}, "
        f"because we can't handle its type {type(schema)}."
    )
