# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections import abc as cabc

__all__ = ["any_set", "AnySet", "any_dict", "AnyDict"]

type AnyId = int

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


class AnySet[K: object = typing.Any]:
    """
    `AnySet` allows to store a set of items, using their `id` or `hash` to compare equality.
    """

    def __init__(self, *, base: type | tuple[type, ...], keys: dict[AnyId, K]) -> None:
        self._keys: dict[AnyId, K] = keys
        """
        The keys that has been stored in the `AnyDict`.
        Using `dict` to avoid actually dereference `id`.
        """

        self._type = base
        """
        Store the type for `isinstance` checks.
        """

    def __repr__(self) -> str:
        return "{" + ", ".join(map(repr, self._keys.values())) + "}"

    def __bool__(self) -> bool:
        return bool(len(self))

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: object, /) -> bool:
        if isinstance(key, self._type):
            key_id = _HASH(key)
            return key_id in self._keys

        raise TypeError(f"{type(key)=} is not `{self._type}`.")

    def __iter__(self) -> cabc.Iterator[K]:
        return iter(self._keys.values())

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
        for item in other:
            self.discard(item)
        return self

    def isdisjoint(self, other: cabc.Iterable[K]) -> bool:
        return self._keys.keys().isdisjoint(_HASH(item) for item in other)

    def add(self, key: K) -> None:
        self._keys[_HASH(key)] = key

    def discard(self, key: K) -> None:
        if (key_hash := _HASH(key)) in self._keys:
            del self._keys[key_hash]

    def copy(self) -> typing.Self:
        return type(self)(base=self._type, keys=self._keys.copy())


class AnyDict[K: object = typing.Any, V: object = typing.Any](AnySet[K]):
    """
    `AnyDict` allows you to treat `T` as if it's `Hashable` (it's not).
    Each item would be compared with `is` rather than `==`.
    """

    def __init__(
        self,
        *,
        base: type | tuple[type, ...],
        keys: dict[AnyId, K],
        vals: dict[AnyId, V],
    ) -> None:
        super().__init__(base=base, keys=keys)

        self._vals: dict[AnyId, V] = vals
        """
        The values refered to by the `key`, using `id` as key.
        """

    def __repr__(self) -> str:
        return "{" + ", ".join(f"{k}:{self[k]}" for k in self) + "}"

    def __getitem__(self, key: K, /) -> V:
        if key not in self:
            raise KeyError(f"{key=} is not found in `AnyDict`.")

        return self._vals[_HASH(key)]

    def __setitem__(self, key: K, val: V, /) -> None:
        self.__assert_same_length()

        super().add(key)
        self._vals[_HASH(key)] = val

    def __delitem__(self, key: K, /) -> None:
        self.__assert_same_length()

        if key not in self:
            raise KeyError(f"{key=} is not in `AnyDict`.")

        super().discard(key)
        del self._vals[_HASH(key)]

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
        assert super().__len__() == len(self._vals)

    # Delete these methods.
    add = discard = typing.cast(typing.Any, None)


# Helper hashing function. ====


@dcls.dataclass
class HashId:
    """
    The hasher that uses `id` for non hashable types.
    """

    no_hash: set[type] = dcls.field(default_factory=set)
    """
    The types that we don't want to hash.
    """

    def __call__(self, obj) -> AnyId:
        if (type_obj := type(obj)) in self.no_hash:
            return id(obj)

        try:
            return hash(obj)

        # Tried hash, but it's not hashable.
        except TypeError:
            self.no_hash.add(type_obj)
            return id(obj)


_HASH = HashId()
"""
The global hasher for both hashable and non hashable objects.
"""
