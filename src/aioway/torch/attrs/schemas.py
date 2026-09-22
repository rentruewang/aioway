# Copyright (c) AIoWay Authors - All Rights Reserved

import json
import typing
from collections import abc as cabc

import loguru as L
import tensordict as td

from aioway._utils import is_tuple_of
from aioway.torch.matches import TorchMatcher

from .attrs import Attr
from .dtypes import DType

__all__ = ["Schema"]


class Schema:
    """
    `Schema` is a `dict[str, Attr]` with additional utilities.
    """

    def __init__(self, mapping: dict[str, typing.Any] | None = None) -> None:
        self._schemas = mapping or {}

    def __contains__(self, key: str | tuple[str, ...]) -> bool:
        if isinstance(key, str):
            return key in self._schemas

        assert is_tuple_of(str)(key) and len(key) > 0, key
        first, *rest = key

        if not rest:
            return first in self

        else:
            return tuple(rest) in self

    def __len__(self) -> int:
        return len(self._schemas)

    def __getitem__(self, key: str | tuple[str, ...], /) -> Attr | Schema:
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
    def get[D](self, key: str | tuple[str, ...], default: D) -> Attr | Schema | D: ...
    @typing.overload
    def get(self, key: str | tuple[str, ...]) -> Attr | Schema | None: ...

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
            assert isinstance(child, Schema)
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

        if not include_nested:
            for key, _ in self._items_of_type(Schema):
                yield key

        # This is the branch where it would yield sub-schema keys as tuples.
        L.logger.trace("Recurse into sub-Schema of {} and yield tuples", self)
        for key, val in self._items_of_type(Schema):
            for child_key in val.keys(include_nested=True):
                child_key = child_key if isinstance(key, tuple) else (child_key,)
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

    @classmethod
    def parse(
        cls, mapping: td.TensorDictBase | cabc.Mapping[str, td.TensorDictBase], /
    ) -> typing.Self:
        parse_child = TorchMatcher(
            tensor=Attr.parse,
            tdict=Schema.parse,
            tcls=Schema.parse,
            mapping=Schema.parse,
        )
        return cls({key: parse_child(tensor) for key, tensor in mapping.items()})
