# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections import abc as cabc

__all__ = ["any_set", "AnySet", "any_dict", "AnyDict"]


# The convenient constructors, public ====


@typing.overload
def any_set() -> AnySet: ...


@typing.overload
def any_set[T](base: type[T], /) -> AnySet[T]: ...


@typing.overload
def any_set[T](base: tuple[type, ...], /) -> AnySet: ...


@typing.overload
def any_set[T](base: typing.Any, /, *default: T) -> AnySet[T]: ...


@typing.overload
def any_set(base, /, *default) -> AnySet: ...


def any_set(base=object, /, *default) -> AnySet:
    aset = AnySet(base=base, keys={})

    # Add all the default values.
    for key in default:
        aset.add(key)

    return aset


@typing.overload
def any_dict() -> AnyDict[typing.Any, typing.Any]: ...


@typing.overload
def any_dict[T](base: type[T], /) -> AnyDict[T, typing.Any]: ...


@typing.overload
def any_dict(base: tuple[type, ...], /) -> AnyDict[typing.Any, typing.Any]: ...


@typing.overload
def any_dict[K, V](base: typing.Any, /, *default: tuple[K, V]) -> AnyDict[K, V]: ...


@typing.overload
def any_dict(base, /, *default) -> AnyDict: ...


def any_dict[K = typing.Any, V = typing.Any](
    base: type | tuple[type, ...] = object, /, *default: tuple[K, V]
):
    adict = AnyDict(base=base, keys={}, vals={})

    # Add all default values.
    for key, val in default:
        adict[key] = val

    return adict


# Class definitions, private. Exposed to `__all__` for type hints. ====


class _AnyBase[K: object = typing.Any]:
    """
    The shared part of `AnySet` and `AnyDict`.
    Keys are stored under `_hashable(key)`.
    """

    def __init__(
        self, *, base: type | tuple[type, ...], keys: dict[cabc.Hashable, K]
    ) -> None:
        self._keys: dict[cabc.Hashable, K] = keys
        """
        The keys that has been stored, under `_hashable(key)`.
        """

        self._type = base
        """
        Store the type for `isinstance` checks.
        """

    def __bool__(self) -> bool:
        return bool(len(self))

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: object, /) -> bool:
        self._check_type(key)
        return _hashable(key) in self._keys

    def __iter__(self) -> cabc.Iterator[K]:
        return iter(self._keys.values())

    def isdisjoint(self, other: cabc.Iterable[K]) -> bool:
        return self._keys.keys().isdisjoint(_hashable(item) for item in other)

    def _check_type(self, key: object) -> None:
        if not isinstance(key, self._type):
            raise TypeError(f"{type(key)=} is not `{self._type}`.")


class AnySet[K: object = typing.Any](_AnyBase[K]):
    """
    `AnySet` allows to store a set of items, using their `id` or `hash` to compare equality.
    """

    def __repr__(self) -> str:
        return "{" + ", ".join(map(repr, self._keys.values())) + "}"

    def __sub__(self, other: AnySet[K]) -> AnySet[K]:
        copied = self.copy()
        copied -= other
        return copied

    def __and__(self, other: AnySet[K]) -> AnySet[K]:
        result = any_set(self._type)
        for item in self:
            if item in other:
                result.add(item)
        return result

    def __isub__(self, other: AnySet[K]) -> AnySet[K]:
        # `list` so that `s -= s` doesn't change `s` while iterating it.
        for item in list(other):
            self.discard(item)
        return self

    def __eq__(self, other) -> bool:
        if not isinstance(other, cabc.Iterable):
            return NotImplemented

        other_keys = {_hashable(elem) for elem in other}
        return self._keys.keys() == other_keys

    def add(self, key: K) -> None:
        self._check_type(key)
        self._keys[_hashable(key)] = key

    def discard(self, key: K) -> None:
        self._keys.pop(_hashable(key), None)

    def copy(self) -> typing.Self:
        return type(self)(base=self._type, keys=self._keys.copy())


class AnyDict[K: object = typing.Any, V: object = typing.Any](_AnyBase[K]):
    """
    `AnyDict` allows you to treat `T` as if it's `Hashable` (it's not).
    Each item would be compared with `is` rather than `==`.
    """

    def __init__(
        self,
        *,
        base: type | tuple[type, ...],
        keys: dict[cabc.Hashable, K],
        vals: dict[cabc.Hashable, V],
    ) -> None:
        super().__init__(base=base, keys=keys)

        self._vals: dict[cabc.Hashable, V] = vals
        """
        The values refered to by the `key`, under `_hashable(key)`.
        """

    def __repr__(self) -> str:
        return "{" + ", ".join(f"{k}:{self[k]}" for k in self) + "}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, (AnyDict, cabc.Mapping)):
            return NotImplemented

        if self.keys() != other.keys():
            return False

        for key, val in self.items():
            if other[key] != val:
                return False

        return True

    def __getitem__(self, key: K, /) -> V:
        if key not in self:
            raise KeyError(f"{key=} is not found in `AnyDict`.")

        return self._vals[_hashable(key)]

    def __setitem__(self, key: K, val: V, /) -> None:
        self.__assert_same_length()
        self._check_type(key)

        hkey = _hashable(key)
        self._keys[hkey] = key
        self._vals[hkey] = val

    def __delitem__(self, key: K, /) -> None:
        self.__assert_same_length()

        if key not in self:
            raise KeyError(f"{key=} is not in `AnyDict`.")

        hkey = _hashable(key)
        del self._keys[hkey]
        del self._vals[hkey]

    def __or__(self, other: typing.Self) -> typing.Self:
        copied = self.copy()
        copied |= other
        return copied

    def __ior__(self, other: typing.Self) -> typing.Self:
        for k, v in other.items():
            self[k] = v
        return self

    def keys(self) -> AnySet[K]:
        return AnySet(base=self._type, keys=self._keys.copy())

    def values(self) -> cabc.Iterable[V]:
        for key in self.keys():
            yield self[key]

    def items(self) -> cabc.Iterable[tuple[K, V]]:
        for key in self.keys():
            yield key, self[key]

    @typing.overload
    def get(self, key: K) -> V | None: ...

    @typing.overload
    def get[D](self, key: K, default: D) -> V | D: ...

    def get(self, key, default=None):
        if key in self:
            return self[key]
        else:
            return default

    def copy(self) -> typing.Self:
        return type(self)(
            base=self._type, keys=self._keys.copy(), vals=self._vals.copy()
        )

    def __assert_same_length(self) -> None:
        assert len(self._keys) == len(self._vals)


# Helpers. ====


@dcls.dataclass(frozen=True, slots=True)
class _Id:
    """
    Wraps an unhashable object so it can be a `dict` key, compared with `is`.
    Never equal to anything but another `_Id` of the same object.
    """

    obj: object

    def __hash__(self):
        return id(self.obj)

    def __eq__(self, other) -> bool:
        return isinstance(other, _Id) and self.obj is other.obj


def _hashable(obj) -> cabc.Hashable:
    "Make the object hashable."

    try:
        _ = hash(obj)
    except Exception:
        return _Id(obj)
    else:
        return obj
