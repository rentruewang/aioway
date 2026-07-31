# Copyright (c) AIoWay Authors - All Rights Reserved

"A collection of tensors and their `Space`s."

import abc
import collections
import dataclasses as dcls
import json
import typing
from collections import abc as cabc

import tensordict as td

from aioway._api import public_api
from aioway._utils import is_real_tensor, torch_fake_mode

from .spaces import Space, space_dcls
from .tensors import Attr, AttrLike, DType

__all__ = ["Schema", "TdictSpace", "TensorClassSpace"]


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

    def to_fake_tensordict(self) -> td.TensorDict:
        with torch_fake_mode():
            return td.TensorDict(
                {key: attr.to_fake_tensor() for key, attr in self.items()}
            )

    @classmethod
    def parse(cls, mapping: cabc.Mapping[str, AttrLike], /) -> typing.Self:
        return cls({key: Attr.parse(tensor) for key, tensor in mapping.items()})


def schema(mapping: cabc.Mapping[str, AttrLike], /) -> Schema:
    return Schema.parse(mapping)


@public_api
@space_dcls
class TensorClassSpace[T: td.TensorClass](Space[T], abc.ABC):
    "A `Space` that checks a `td.TensorClass`."

    KLASS: typing.ClassVar[type[T]]
    "The class for which to check."

    def __post_init__(self) -> None:
        if not isinstance(self.KLASS, type):
            raise TypeError(f"{self.KLASS=} should be a type.")

        if not issubclass(self.KLASS, td.TensorClass):
            raise TypeError(f"{self.KLASS=} should be a subclass of `td.TensorClass`.")

    @typing.final
    def contains(self, inst: T) -> bool:
        if not isinstance(inst, self.KLASS):
            raise TypeError(
                f"{inst=} should be an instance of `{self.KLASS!s}`. "
                f"But {type(inst)=}."
            )

        schema = self._to_schema(inst)

        try:
            self._check_attrs(schema)
        except ValueError:
            return False

        try:
            self._check_data(inst)
        except ValueError:
            return False

        return True

    @typing.no_type_check
    def _to_schema(self, inst: T) -> Schema:
        assert dcls.is_dataclass(inst)
        fields = dcls.asdict(inst)
        schema = Schema.parse(fields)
        return schema

    @abc.abstractmethod
    def _check_attrs(self, attrs: Schema, /) -> None:
        """
        Raise `ValueError` if `self` is incompatible with tdict with `attrs`.
        """

    @abc.abstractmethod
    def _check_data(self, inst: T, /) -> None:
        """
        Raise `ValueError` if `inst` is not acceptable.
        """


@public_api
@space_dcls
class TdictSpace(Space[td.TensorDict]):
    "A `Space` that checks a `td.TensorDict`."

    @typing.override
    @typing.final
    def contains(self, tdict: td.TensorDict, /) -> bool:
        if not isinstance(tdict, td.TensorDict):
            raise TypeError(f"{type(tdict)} is not a `td.TensorDict`.")

        attrs = Schema.parse(tdict)

        try:
            self._check_attrs(attrs)

            # Only perform the data checks if all the values are real.
            if all(map(is_real_tensor, tdict.values())):
                self._check_data(tdict)
        except ValueError:
            return False
        else:
            return True

    @abc.abstractmethod
    def _check_attrs(self, attrs: Schema, /) -> None:
        """
        Raise `ValueError` if `self` is incompatible with tdict with `attrs`.
        """

    @abc.abstractmethod
    def _check_data(self, tdict: td.TensorDict, /) -> None:
        """
        Raise `ValueError` if `self` is not valid or is incompatible with `tdict`.
        """
