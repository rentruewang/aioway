# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing

from aioway._ufuncs import UFunc
from aioway._utils import Sign
from collections import abc as cabc
from .spaces import Space

__all__ = ["register_cast", "cast_space", "Caster", "CastedUFunc"]

_CASTERS: dict[tuple[type[Space], type[Space]], Caster] = {}
"The casters organized by `tuple[type[Space], type[Space]]`."


@dcls.dataclass(frozen=True)
class CastedUFunc[S: Space = Space, T: Space = Space](UFunc):
    """
    `CastedUFunc` outputs the space,
    and the corresponding `UFunc` that would generate valid outputs.
    """

    func: cabc.Callable[[typing.Any], typing.Any]
    "The corresponding encoder."

    in_space: S
    "The space that is converted from."

    out_space: T
    "The converted space."

    @typing.override
    def forward(self, item: typing.Any) -> typing.Any:
        return self.func(item)


@typing.runtime_checkable
class Caster[S: Space, T: Space](typing.Protocol):
    """
    `Caster` defines a possible cast from input to output,
    where the input is in the `input` `Space`, and output is in the `output` `Space`.
    """

    @abc.abstractmethod
    def __call__(self, observ_space_type: S, /) -> CastedUFunc[T]:
        raise NotImplementedError


def register_cast[C: Caster](cast: C) -> C:
    signature = Sign.from_callable(cast)

    assert len(signature.parameters) == 1
    input_space_type = next(
        iter(Sign.from_callable(cast).parameters.values())
    ).annotation
    output_space_type = typing.get_args(signature.return_annotation)[0]
    in_out_types = input_space_type, output_space_type

    if in_out_types in _CASTERS:
        raise KeyError(
            f"{in_out_types} already exits. Current: {_CASTERS[in_out_types]}."
        )

    _CASTERS[in_out_types] = cast
    return cast


def cast_space[S: Space, T: Space](space: S, target: type[T]) -> CastedUFunc[S, T]:
    """
    Cast `space`, a `Space` instance, to another space of type `target`.

    If the cast function is not found, `NotImplemented` is returned.
    """

    input_type = type(space)
    cast = _CASTERS[input_type, target]
    return
    return cast(space)


def _is_space_type(obj) -> typing.TypeIs[type[Space]]:
    return isinstance(obj, type) and issubclass(obj, Space)
