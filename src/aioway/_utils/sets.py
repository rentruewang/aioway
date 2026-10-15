# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections import abc as cabc

__all__ = [
    "any_set",
    "AnySet",
    "any_dict",
    "AnyDict",
    "AnyKey",
    "HashableKey",
    "UnhashableKey",
    "wrap_any",
]


# The convenient constructors, public ====


@typing.overload
def any_set() -> AnySet: ...


@typing.overload
def any_set[T](*default: T) -> AnySet[T]: ...


@typing.overload
def any_set(*default) -> AnySet: ...


def any_set(*default) -> AnySet:
    aset = AnySet(data={})

    # Add all the default values.
    for key in default:
        aset.add(key)

    return aset


@typing.overload
def any_dict() -> AnyDict[typing.Any, typing.Any]: ...


@typing.overload
def any_dict[K, V](*default: tuple[K, V]) -> AnyDict[K, V]: ...


def any_dict[K, V](*default: tuple[K, V]):
    adict = AnyDict(data={})

    # Add all default values.
    for key, val in default:
        adict[key] = val

    return adict


# Class definitions. ====


class AnySet[K: object = typing.Any]:
    """
    `AnySet` allows to store a set of any items.
    Hashable items are compared with `hash` and `==`, unhashable ones with `is`.
    """

    def __init__(self, *, data: dict[AnyKey[K], None]) -> None:
        self._data: dict[AnyKey[K], None] = data
        """
        The wrapped items. A `dict` rather than a `set`, to keep insertion order.
        """

    def __repr__(self) -> str:
        return "{" + ", ".join(map(repr, self)) + "}"

    def __bool__(self) -> bool:
        return bool(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object, /) -> bool:
        return wrap_any(key) in self._data

    def __iter__(self) -> cabc.Iterator[K]:
        return (key.obj for key in self._data)

    def __eq__(self, other) -> bool:
        if isinstance(other, AnySet):
            return self._data.keys() == other._data.keys()

        if not isinstance(other, cabc.Iterable):
            return NotImplemented

        return self._data.keys() == {wrap_any(elem) for elem in other}

    def __sub__(self, other: AnySet[K]) -> AnySet[K]:
        data = {key: None for key in self._data if key not in other._data}
        return type(self)(data=data)

    def __and__(self, other: AnySet[K]) -> AnySet[K]:
        data = {key: None for key in self._data if key in other._data}
        return type(self)(data=data)

    def __isub__(self, other: AnySet[K]) -> AnySet[K]:
        # `list` so that `s -= s` doesn't change `s` while iterating it.
        for key in list(other._data):
            self._data.pop(key, None)
        return self

    def isdisjoint(self, other: cabc.Iterable[K]) -> bool:
        return self._data.keys().isdisjoint(wrap_any(item) for item in other)

    def add(self, key: K) -> None:
        self._data[wrap_any(key)] = None

    def discard(self, key: K) -> None:
        self._data.pop(wrap_any(key), None)

    def copy(self) -> typing.Self:
        return type(self)(data=self._data.copy())


class AnyDict[K, V]:
    """
    `AnyDict` allows you to treat `K` as if it's `Hashable` (it may not be).
    Hashable keys are compared with `hash` and `==`, unhashable ones with `is`.
    """

    def __init__(self, *, data: dict[AnyKey[K], V]) -> None:
        self._data: dict[AnyKey[K], V] = data
        """
        The wrapped keys, mapped to their values.
        """

    def __repr__(self) -> str:
        return "{" + ", ".join(f"{k}:{v}" for k, v in self.items()) + "}"

    def __bool__(self) -> bool:
        return bool(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object, /) -> bool:
        return wrap_any(key) in self._data

    def __iter__(self) -> cabc.Iterator[K]:
        return (key.obj for key in self._data)

    def __eq__(self, other) -> bool:
        # Both sides keyed by `wrap_any(key)`: `dict.__eq__` does the rest,
        # comparing values with `is`, then `==`.
        if isinstance(other, AnyDict):
            return self._data == other._data

        if isinstance(other, cabc.Mapping):
            return self._data == {wrap_any(k): v for k, v in other.items()}

        return NotImplemented

    def __getitem__(self, key: K, /) -> V:
        try:
            return self._data[wrap_any(key)]
        except KeyError:
            raise KeyError(f"{key=} is not found in `AnyDict`.") from None

    def __setitem__(self, key: K, val: V, /) -> None:
        self._data[wrap_any(key)] = val

    def __delitem__(self, key: K, /) -> None:
        try:
            del self._data[wrap_any(key)]
        except KeyError:
            raise KeyError(f"{key=} is not in `AnyDict`.") from None

    def __or__(self, other: typing.Self) -> typing.Self:
        copied = self.copy()
        copied |= other
        return copied

    def __ior__(self, other: typing.Self) -> typing.Self:
        for k, v in other.items():
            self[k] = v
        return self

    def keys(self) -> AnySet[K]:
        data: dict[AnyKey[K], None] = dict.fromkeys(self._data)
        return AnySet(data=data)

    def values(self) -> cabc.Iterable[V]:
        return list(self._data.values())

    def items(self) -> cabc.Iterable[tuple[K, V]]:
        return [(key.obj, val) for key, val in self._data.items()]

    @typing.overload
    def get(self, key: K) -> V | None: ...

    @typing.overload
    def get[D](self, key: K, default: D) -> V | D: ...

    def get(self, key, default=None):
        return self._data.get(wrap_any(key), default)

    def copy(self) -> typing.Self:
        return type(self)(data=self._data.copy())


# Helpers. ====


class AnyKey[T = object](typing.Protocol):
    "The keys used in `AnySet` and `AnyDict`."

    def __hash__(self) -> int: ...

    def __eq__(self, other) -> bool: ...

    @property
    def obj(self) -> T: ...


@dcls.dataclass(frozen=True, slots=True)
class UnhashableKey[T = object]:
    """
    Wraps an unhashable object to be hashed, comparing with `is`.
    """

    obj: T

    def __hash__(self) -> int:
        return id(self.obj)

    def __eq__(self, other) -> bool:
        return isinstance(other, UnhashableKey) and self.obj is other.obj


@dcls.dataclass(frozen=True, slots=True)
class HashableKey[T = object]:
    """
    Wraps a hashable object to avoid recomputation.

    It delegates `==` to be `hash == hash` and `is` or `==` on the object.
    """

    obj: T

    key: int

    def __hash__(self) -> int:
        return self.key

    def __eq__(self, other) -> bool:
        if not isinstance(other, HashableKey):
            return False

        if self.key != other.key:
            return False

        if self.obj is other.obj:
            return True

        if self.obj == other.obj:
            return True

        return False


def wrap_any[T = object](obj: T) -> AnyKey[T]:
    "Convenient function to wrap the object as an internal key."

    try:
        key = hash(obj)
    except Exception:
        return UnhashableKey(obj)
    else:
        return HashableKey(obj, key)
