# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

from torch import nn

from .spaces import Space

__all__ = ["register_cast", "cast_space", "Caster", "CastedModule", "CastedSpaceModule"]

_CASTERS: dict[tuple[type[Space], type[Space]], Caster] = {}
"The casters organized by `tuple[type[Space], type[Space]]`."


@dcls.dataclass(frozen=True)
class CastedModule[S: Space = Space, T: Space = Space](nn.Module):
    """
    `CastedModule` outputs the space,
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


class CastedSpaceModule[S: Space = Space](typing.NamedTuple):
    "The tuple with the output cast space and the module."

    space: S
    "The space that is casted."

    module: nn.Module
    "The function that would be used for conversion."


@typing.runtime_checkable
class Caster[S: Space, T: Space](typing.Protocol):
    """
    `Caster` defines a possible cast from input to output,
    where the input is in the `input` `Space`, and output is in the `output` `Space`.
    """

    @abc.abstractmethod
    def __call__(self, observ_space_type: S, /) -> CastedSpaceModule[T]:
        raise NotImplementedError


def register_cast(input_type, output_type):
    if not _is_space_type(input_type):
        raise TypeError(f"{input_type=} is not a `type[Space]`.")
    if not _is_space_type(output_type):
        raise TypeError(f"{output_type=} is not a `type[Space]`.")

    def decorator[C: Caster](cast: C) -> C:
        in_out_types = input_type, output_type

        # Do the register here.
        if in_out_types in _CASTERS:
            raise KeyError(
                f"{in_out_types} already exits. Current: {_CASTERS[in_out_types]}."
            )
        _CASTERS[in_out_types] = cast

        return cast

    return decorator


def cast_space(space: Space, target_type: type[Space], /) -> CastedModule:
    """
    Cast `space`, a `Space` instance, to another space of type `target`.

    If the cast function is not found, `NotImplemented` is returned.
    """

    input_type = type(space)
    cast = _CASTERS[input_type, target_type]
    target_space, module = cast(space)
    return CastedModule(module, space, target_space)


def _is_space_type(obj) -> typing.TypeIs[type[Space]]:
    return isinstance(obj, type) and issubclass(obj, Space)
