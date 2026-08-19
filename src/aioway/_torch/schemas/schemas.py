# Copyright (c) AIoWay Authors - All Rights Reserved

import collections
import json
import typing
from collections import abc as cabc

import tensordict as td

from .attrs import Attr, AttrLike
from .dtypes import DType

__all__ = ["Schema"]


class Schema(collections.UserDict[str, Attr]):
    """
    `Schema` is a `dict[str, Attr]` with additional utilities.
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
        Renames the current `Schema`.
        """

        return type(self)({renames.get(key, key): val for key, val in self.items()})

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

    def to_fake_tdict(self) -> td.TensorDict:
        from ..fake import fake_mode

        with fake_mode():
            return td.TensorDict(
                {key: attr.to_fake_tensor() for key, attr in self.items()}
            )

    @classmethod
    def parse(cls, mapping: cabc.Mapping[str, AttrLike], /) -> typing.Self:
        return cls({key: Attr.parse(tensor) for key, tensor in mapping.items()})


def schema(mapping: cabc.Mapping[str, AttrLike], /) -> Schema:
    return Schema.parse(mapping)
