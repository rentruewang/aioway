# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

__all__ = ["AnySet", "AnyDict"]

type AnyId = int


class AnySet[K = typing.Any]:
    """
    `AnySet` allows to store a set of items, using their `id` or `hash` to compare equality.
    """

    def __init__(self, base: type | tuple[type, ...] = object, *default: K) -> None:
        self.__keys: dict[AnyId, K] = {}
        """
        The keys that has been stored in the `AnyDict`.
        Using `dict` to avoid actually dereference `id`.
        """

        self.__type = base
        "Store the type for `isinstance` checks."

        # Add all the default values.
        for val in default:
            self.add(val)

    def __repr__(self) -> str:
        return "{" + ", ".join(map(repr, self.__keys.values())) + "}"

    def __bool__(self) -> bool:
        return bool(len(self))

    def __len__(self) -> int:
        return len(self.__keys)

    def __contains__(self, key: object, /) -> bool:
        if isinstance(key, self.__type):
            key_id = _hash_or_id(key)
            return key_id in self.__keys

        raise TypeError(f"{type(key)=} is not `{self.__type}`.")

    def __iter__(self) -> cabc.Iterator[K]:
        yield from self.__keys.values()

    def add(self, key: K) -> None:
        self.__keys[_hash_or_id(key)] = key

    def discard(self, key: K) -> None:
        if (key_hash := _hash_or_id(key)) in self.__keys:
            del self.__keys[key_hash]


class AnyDict[K = typing.Any, V = typing.Any](AnySet[K]):
    """
    `AnyDict` allows you to treat `T` as if it's `Hashable` (it's not).
    Each item would be compared with `is` rather than `==`.
    """

    def __init__(
        self, base: type | tuple[type, ...] = object, *default: tuple[K, V]
    ) -> None:
        super().__init__(base)

        self.__vals: dict[int, V] = {}
        """
        The values refered to by the `key`, using `id` as key.
        """

        for key, val in default:
            self[key] = val

    def __repr__(self) -> str:
        return "{" + ", ".join(f"{k}:{self[k]}" for k in self) + "}"

    def __getitem__(self, key: K, /) -> V:
        if key not in self:
            raise KeyError(f"{key=} is not found in `AnyDict`.")

        return self.__vals[_hash_or_id(key)]

    def __setitem__(self, key: K, val: V, /) -> None:
        self.__assert_same_length()

        super().add(key)
        self.__vals[_hash_or_id(key)] = val

    def __delitem__(self, key: K, /) -> None:
        self.__assert_same_length()

        if key not in self:
            raise KeyError(f"{key=} is not in `AnyDict`.")

        super().discard(key)
        del self.__vals[_hash_or_id(key)]

    def keys(self) -> cabc.KeysView[K]:
        return cabc.KeysView(self)

    def values(self) -> cabc.ValuesView[V]:
        return cabc.ValuesView(self)

    def items(self) -> cabc.ItemsView[K, V]:
        return cabc.ItemsView(self)

    @typing.overload
    def get(self, key: K) -> V | None: ...

    @typing.overload
    def get[D](self, key: K, default: D) -> V | D: ...

    def get(self, key, default=None):
        if key in self:
            return self[key]
        else:
            return default

    def __assert_same_length(self) -> None:
        assert super().__len__() == len(self.__vals)

    # Delete these methods.
    add = discard = typing.cast(typing.Any, None)


def _hash_or_id(obj) -> AnyId:
    "Get the hash or id values."
    return hash(obj) if isinstance(obj, cabc.Hashable) else id(obj)
