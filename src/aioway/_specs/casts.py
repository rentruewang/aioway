# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

from torch import nn
from torchrl import data as rldata

__all__ = ["register_cast", "cast_space", "Caster", "CastedModule", "CastedSpaceModule"]

_CASTERS: dict[tuple[type[rldata.TensorSpec], type[rldata.TensorSpec]], Caster] = {}
"The casters organized by `tuple[type[rldata.TensorSpec], type[rldata.TensorSpec]]`."


@dcls.dataclass(frozen=True)
class CastedModule[
    S: rldata.TensorSpec = rldata.TensorSpec, T: rldata.TensorSpec = rldata.TensorSpec
](nn.Module):
    """
    `CastedModule` outputs the `rldata.TensorSpec`,
    and the corresponding `nn.Module` that would generate valid outputs.
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


class CastedSpaceModule[S: rldata.TensorSpec = rldata.TensorSpec](typing.NamedTuple):
    "The tuple with the output cast space and the module."

    space: S
    "The space that is casted."

    module: nn.Module
    "The function that would be used for conversion."


@typing.runtime_checkable
class Caster[S: rldata.TensorSpec, T: rldata.TensorSpec](typing.Protocol):
    """
    `Caster` defines a possible cast from input to output,
    where the input is in the `input` `rldata.TensorSpec`,
    and output is in the `output` `rldata.TensorSpec`.
    """

    @abc.abstractmethod
    def __call__(self, observ_space_type: S, /) -> CastedSpaceModule[T]:
        raise NotImplementedError


def register_cast(input_type, output_type):
    if not _is_tensor_spec_type(input_type):
        raise TypeError(f"{input_type=} is not a `type[rldata.TensorSpec]`.")
    if not _is_tensor_spec_type(output_type):
        raise TypeError(f"{output_type=} is not a `type[rldata.TensorSpec]`.")

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


def cast_space(
    space: rldata.TensorSpec, target_type: type[rldata.TensorSpec], /
) -> CastedModule:
    """
    Cast `space`, a `rldata.TensorSpec` instance, to another space of type `target`.

    If the cast function is not found, `NotImplemented` is returned.
    """

    input_type = type(space)
    cast = _CASTERS[input_type, target_type]
    target_space, module = cast(space)
    return CastedModule(module, space, target_space)


def _is_tensor_spec_type(obj) -> typing.TypeIs[type[rldata.TensorSpec]]:
    return isinstance(obj, type) and issubclass(obj, rldata.TensorSpec)
