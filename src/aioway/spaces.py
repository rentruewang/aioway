# Copyright (c) AIoWay Authors - All Rights Reserved

"A `Space` is a constraint on `Schema`, used by an `Algo` to constrain `Hop`."

import abc
import dataclasses as dcls
import typing

from .schemas import Schema

__all__ = ["space_dcls", "Space", "AnySpace"]


@typing.dataclass_transform(frozen_default=True)
def space_dcls(cls):
    return dcls.dataclass(frozen=True, slots=True)(cls)


@space_dcls
class Space(abc.ABC):
    """
    The base class for `Space`.
    """

    @abc.abstractmethod
    def __contains__(self, schema: Schema, /) -> bool:
        """
        If a `schema` lies in a `Space`, `in` would be `True`.
        """

        raise NotImplementedError


@space_dcls
class AnySpace(Space):
    """
    No constraint on the input `Schema`.
    """

    @typing.override
    def __contains__(self, schema: Schema, /) -> bool:
        return True


@space_dcls
class DiscreteSpace(Space):
    "Discrete space that allows for arbitrary integers."

    ndim: int
    """
    The number of dimensions of a discrete space. Must be >= 0.
    """

    @typing.override
    def __contains__(self, schema: Schema, /) -> bool:
        return schema.shape.ndim == self.ndim and schema.dtype.family == "int"


@space_dcls
class ContinuousSpace(Space):
    "Continuous space that allows for arbitrary floating point numbers."

    ndim: int
    """
    The number of dimensions of a discrete space. Must be >= 0.
    """

    @typing.override
    def __contains__(self, schema: Schema, /) -> bool:
        return schema.shape.ndim == self.ndim and schema.dtype.is_floating_point
