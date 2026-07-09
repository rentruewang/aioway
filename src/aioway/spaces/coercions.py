# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import abc
import dataclasses as dcls
import typing

from aioway._ufuncs import UFunc

from .spaces import Space

__all__ = ["register_coercsion", "coerce_space", "Coercion", "CoercionOutput"]


@dcls.dataclass(frozen=True)
class CoercionOutput[S: Space = Space]:
    """
    `Coercion` outputs the space,
    and the corresponding `UFunc` that would generate valid outputs.
    """

    out_space: S
    "The converted space."

    ufunc: UFunc
    "The corresponding encoder."


class Coercion[S: Space, T: Space](typing.Protocol):
    """
    `Coercion` defines a possible coercion from input to output,
    where the input is in the `input` `Space`, and output is in the `output` `Space`.
    """

    @abc.abstractmethod
    def __call__(self, observ_space_type: S, /) -> CoercionOutput[T]:
        raise NotImplementedError


@typing.runtime_checkable
class SpaceCoercion[S: Space, T: Space](typing.Protocol):
    def __call__(self, space: S, /) -> T: ...


def register_coercsion[C: Coercion](coercion: C) -> C:
    return coercion


def coerce_space[S: Space, T: Space](space: S, target: type[T]) -> T:
    """
    Cast `space`, a `Space` instance, to another space of type `target`.

    If the coercion function is not found, `NotImplemented` is returned.
    """

    pytest.xfail("Fail")


def _is_space_type(obj) -> typing.TypeIs[type[Space]]:
    return isinstance(obj, type) and issubclass(obj, Space)
