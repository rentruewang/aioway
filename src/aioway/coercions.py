# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing

from aioway._ufuncs import UFunc
from aioway._utils import Sign

if typing.TYPE_CHECKING:
    from aioway.spaces import Space

__all__ = ["register_coercsion", "coerce_space", "Coercion", "CoercionOutput"]

_COERCIONS: dict[tuple[type[Space], type[Space]], Coercion] = {}


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
    signature = Sign.from_callable(coercion)

    assert len(signature.parameters) == 1
    input_space_type = next(
        iter(Sign.from_callable(coercion).parameters.values())
    ).annotation
    output_space_type = typing.get_args(signature.return_annotation)[0]
    in_out_types = input_space_type, output_space_type

    if in_out_types in _COERCIONS:
        raise KeyError(
            f"{in_out_types} already exits. Current: {_COERCIONS[in_out_types]}."
        )

    _COERCIONS[in_out_types] = coercion
    return coercion


def coerce_space[S: Space, T: Space](space: S, target: type[T]) -> CoercionOutput:
    """
    Cast `space`, a `Space` instance, to another space of type `target`.

    If the coercion function is not found, `NotImplemented` is returned.
    """

    input_type = type(space)
    coercion = _COERCIONS[input_type, target]
    return coercion(space)


def _is_space_type(obj) -> typing.TypeIs[type[Space]]:
    return isinstance(obj, type) and issubclass(obj, Space)
