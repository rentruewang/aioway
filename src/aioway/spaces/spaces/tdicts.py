# Copyright (c) AIoWay Authors - All Rights Reserved

"A collection of tensors and their `Space`s."

import abc
import dataclasses as dcls
import typing

import tensordict as td

from aioway._api import public_api
from aioway._torch import Schema, is_real_tensor

from .spaces import Space, space_dcls

__all__ = ["Schema", "TdictSpace", "TClsSpace"]


@public_api
@space_dcls
class TClsSpace[T: td.TensorClass](Space[T], abc.ABC):
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
