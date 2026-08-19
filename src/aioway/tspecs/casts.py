# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

from torch import nn

from .tspecs import TSpec

__all__ = ["register_cast", "cast_tspec", "Caster", "CastedModule", "CastedSpaceModule"]

_CASTERS: dict[tuple[type[TSpec], type[TSpec]], Caster] = {}
"The casters organized by `tuple[type[TSpec], type[TSpec]]`."


@dcls.dataclass(frozen=True)
class CastedModule[S: TSpec = TSpec, T: TSpec = TSpec](nn.Module):
    """
    `CastedModule` outputs the `TSpec`,
    and the corresponding `nn.Module` that would generate valid outputs.
    """

    func: cabc.Callable[[typing.Any], typing.Any]
    "The corresponding encoder."

    in_tspec: S
    "The tspec that is converted from."

    out_tspec: T
    "The converted tspec."

    @typing.override
    def forward(self, item: typing.Any) -> typing.Any:
        return self.func(item)


class CastedSpaceModule[S: TSpec = TSpec](typing.NamedTuple):
    "The tuple with the output cast space and the module."

    tspec: S
    "The tspec that is casted."

    module: nn.Module
    "The function that would be used for conversion."


@typing.runtime_checkable
class Caster[S: TSpec, T: TSpec](typing.Protocol):
    """
    `Caster` defines a possible cast from input to output,
    where the input is in the `input` `TSpec`,
    and output is in the `output` `TSpec`.
    """

    @abc.abstractmethod
    def __call__(self, observ_space_type: S, /) -> CastedSpaceModule[T]:
        raise NotImplementedError


def register_cast(input_type, output_type):
    if not _is_tensor_spec_type(input_type):
        raise TypeError(f"{input_type=} is not a `type[TSpec]`.")
    if not _is_tensor_spec_type(output_type):
        raise TypeError(f"{output_type=} is not a `type[TSpec]`.")

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


def cast_tspec(tspec: TSpec, target_type: type[TSpec], /) -> CastedModule:
    """
    Cast `tspec`, a `TSpec` instance, to another space of type `target`.

    If the cast function is not found, `NotImplemented` is returned.
    """

    input_type = type(tspec)
    cast = _CASTERS[input_type, target_type]
    target_tspec, module = cast(tspec)
    return CastedModule(module, tspec, target_tspec)


def _is_tensor_spec_type(obj) -> typing.TypeIs[type[TSpec]]:
    return isinstance(obj, type) and issubclass(obj, TSpec)
