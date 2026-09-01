# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import inspect
import typing
from collections import abc as cabc

import numpy as np
from numpy import typing as npt

__all__ = [
    "IntArray",
    "IntArrayLike",
    "UIntArray",
    "FloatArray",
    "FloatArrayLike",
    "BoolArray",
    "BoolArrayLike",
    "is_list_of",
    "is_tuple_of",
    "is_seq_of",
    "is_any_type_hint",
    "is_dict_of_str_to",
    "HasLen",
    "SeqKeysView",
    "SetKeysView",
]

type IntArray = npt.NDArray[np.int_]
type IntArrayLike = tuple[int, ...] | list[int] | IntArray
type FloatArray = npt.NDArray[np.floating]
type FloatArrayLike = tuple[float, ...] | list[float] | FloatArray
type BoolArray = npt.NDArray[np.bool_]
type BoolArrayLike = tuple[bool, ...] | list[bool] | BoolArray
type UIntArray = npt.NDArray[np.uint]


_ANY_TYPE = inspect.Parameter.empty, typing.Any, object


def is_any_type_hint(typ) -> bool:
    "Check if `typ` is no constraint."
    return isinstance(typ, type) and typ in _ANY_TYPE


@typing.no_type_check
def _seq_check[T](seq: type, typ: type[T]):
    if not issubclass(seq, cabc.Sequence):
        raise TypeError(
            f"The given seq: `{seq}` should be subclass of `cabc.Sequence`."
        )

    if not isinstance(typ, type):
        raise TypeError(f"The given typ: `{typ}` should be a type.")

    def check(obj) -> typing.TypeGuard[typing.Any]:
        return isinstance(obj, seq) and all(isinstance(i, typ) for i in obj)

    return check


@typing.no_type_check
def _mapping_check[K, V](mapping: type, key: type[K], val: type[V]):
    if not issubclass(mapping, cabc.Mapping):
        raise TypeError(
            f"The given mapping: `{mapping}` should be subclass of `cabc.Mapping`."
        )

    if not isinstance(key, type):
        raise TypeError(f"The given key: `{key}` should be a type.")

    if not isinstance(val, type):
        raise TypeError(f"The given val: `{val}` should be a type.")

    def check(obj) -> typing.TypeGuard[typing.Any]:
        return isinstance(obj, mapping) and all(
            isinstance(k, key) and isinstance(v, val) for k, v in obj.items()
        )

    return check


def is_seq_of[T](
    typ: type[T], /
) -> cabc.Callable[[typing.Any], typing.TypeGuard[cabc.Sequence[T]]]:
    return _seq_check(cabc.Sequence, typ)


def is_list_of[T](
    typ: type[T], /
) -> cabc.Callable[[typing.Any], typing.TypeGuard[list[T]]]:
    return _seq_check(list, typ)


def is_tuple_of[T](
    typ: type[T], /
) -> cabc.Callable[[typing.Any], typing.TypeGuard[tuple[T, ...]]]:
    return _seq_check(tuple, typ)


def is_dict_of_str_to[T](
    typ: type[T], /
) -> cabc.Callable[[typing.Any], typing.TypeGuard[dict[str, T]]]:
    return _mapping_check(dict, str, typ)


@typing.runtime_checkable
class HasLen(typing.Protocol):
    def __len__(self) -> int: ...


@dcls.dataclass(frozen=True)
class _ContainerKeysView[C: cabc.Sequence[str] | cabc.Set[str]](cabc.KeysView[str]):
    seq: C

    @typing.override
    def __contains__(self, key: object) -> bool:
        return key in self.seq

    @typing.override
    def __iter__(self):
        yield from self.seq


class SeqKeysView(_ContainerKeysView[cabc.Sequence[str]]): ...


class SetKeysView(_ContainerKeysView[set[str]]): ...
