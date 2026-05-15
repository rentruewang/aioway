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
from aioway.schemas import LayoutLike

from .attrs import Attr, attr
from .devices import DeviceLike
from .dtypes import DTypeLike
from .shapes import ShapeLike

__all__ = ["AttrSet", "AttrSetLike", "attr_set"]


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
class AttrSet(cabc.Mapping[str, Attr]):
    """
    The collection of `Attr`s. This is the data type for a `td.TensorDict`.

    Right now the columns are in sorted order, but this is not guarenteed.
    Most likely will change in the future.
    """

    attr_items: tuple[_AttrItem[Attr], ...] = ()
    """
    The data backing the `AttrSet`. Must be sorted.
    """

    def __post_init__(self) -> None:
        if len(self.names) > 1 and not np.all(self.names[:-1] <= self.names[1:]):
            raise ValueError(f"Names are not sorted: {self.names}.")

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

        raise KeyError(f"AttrSet cannot handle {idx=}.")

    @typing.override
    def __repr__(self) -> str:
        return self._repr_string

    def __or__(self, other: cabc.Mapping[str, Attr]) -> typing.Self:
        return type(self).from_dict({**self, **other})

    def __len__(self) -> int:
        return len(self.attr_items)

    def __iter__(self) -> cabc.Iterator[str]:
        return (attr.name for attr in self.attr_items)

    def __getitem_str(self, idx: str):
        return self.column(idx)

    def __getitem_list_str(self, idx: list[str]):
        return self.select(*idx)

    def rename(self, **renames: str):
        return self.from_dict(
            {renames.get(key, key): attr for key, attr in self.items()}
        )

    def _get_attr_list[T](self, get: cabc.Callable[[Attr], T]):
        return [get(col.attr) for col in self.attr_items]

    def keys(self):
        return self._keys_view

    def column(self, key: str, /) -> Attr:
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
    def from_values(cls, **mapping: Attr) -> typing.Self:
        return cls.from_dict(mapping)

    @classmethod
    def from_fields(
        cls,
        *,
        names: cabc.Sequence[str],
        shape_list: cabc.Sequence[ShapeLike],
        dtype_list: cabc.Sequence[DTypeLike],
        device_list: cabc.Sequence[DeviceLike],
        layout_list: cabc.Sequence[LayoutLike],
        requires_grad_list: cabc.Sequence[bool],
    ) -> typing.Self:
        "Create an `AttrSet` from a set of seuqences of attributes of same length."

        lengths = {
            len(names),
            len(shape_list),
            len(dtype_list),
            len(device_list),
            len(layout_list),
            len(requires_grad_list),
        }

        if not len(lengths) == 1:
            raise ValueError(
                "Should all have the same length. "
                f"Got {len(names)=}, {len(shape_list)=}, {len(dtype_list)=}, "
                f"{len(device_list)=}, {len(layout_list)=}, {len(requires_grad_list)=}."
            )

        mapping: dict[str, Attr] = {}
        for i, name in enumerate(names):
            item = attr(
                {
                    "device": device_list[i],
                    "dtype": dtype_list[i],
                    "shape": shape_list[i],
                    "layout": layout_list[i],
                    "requires_grad": requires_grad_list[i],
                }
            )
            mapping[name] = item

        return cls.from_dict(mapping)

    @classmethod
    def from_tensordict(cls, data: td.TensorDict, /) -> typing.Self:
        return cls.from_dict({key: attr(val) for key, val in data.items()})

    @classmethod
    def from_dict(cls, mapping: cabc.Mapping[str, Attr], /) -> typing.Self:
        return cls(
            tuple(
                sorted(
                    _AttrItem(name=name, attr=attr) for name, attr in mapping.items()
                )
            )
        )


@dcls.dataclass(frozen=True)
class _AttrKeysView(cabc.KeysView[str]):
    attrset: AttrSet
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
