# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

from torch import nn
from torchrl.data import tensor_specs as tspecs

__all__ = ["register_cast", "cast_tspec", "Caster", "CastedModule", "CastedSpaceModule"]

_CASTERS: dict[tuple[type[tspecs.TensorSpec], type[tspecs.TensorSpec]], Caster] = {}
"The casters organized by `tuple[type[tspecs.TensorSpec], type[tspecs.TensorSpec]]`."


@dcls.dataclass(frozen=True)
class CastedModule[
    S: tspecs.TensorSpec = tspecs.TensorSpec, T: tspecs.TensorSpec = tspecs.TensorSpec
](nn.Module):
    """
    `CastedModule` outputs the `tspecs.TensorSpec`,
    and the corresponding `nn.Module` that would generate valid outputs.
    """

    func: cabc.Callable[[typing.Any], typing.Any]
    "The corresponding encoder."

    in_tspec: S
    "The space that is converted from."

    out_tspec: T
    "The converted space."

    @typing.override
    def forward(self, item: typing.Any) -> typing.Any:
        return self.func(item)


class CastedSpaceModule[S: tspecs.TensorSpec = tspecs.TensorSpec](typing.NamedTuple):
    "The tuple with the output cast space and the module."

    space: S
    "The space that is casted."

    module: nn.Module
    "The function that would be used for conversion."


@typing.runtime_checkable
class Caster[S: tspecs.TensorSpec, T: tspecs.TensorSpec](typing.Protocol):
    """
    `Caster` defines a possible cast from input to output,
    where the input is in the `input` `tspecs.TensorSpec`,
    and output is in the `output` `tspecs.TensorSpec`.
    """

    @abc.abstractmethod
    def __call__(self, observ_space_type: S, /) -> CastedSpaceModule[T]:
        raise NotImplementedError


def register_cast(input_type, output_type):
    if not _is_tensor_spec_type(input_type):
        raise TypeError(f"{input_type=} is not a `type[tspecs.TensorSpec]`.")
    if not _is_tensor_spec_type(output_type):
        raise TypeError(f"{output_type=} is not a `type[tspecs.TensorSpec]`.")

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


def cast_tspec(
    space: tspecs.TensorSpec, target_type: type[tspecs.TensorSpec], /
) -> CastedModule:
    """
    Cast `space`, a `tspecs.TensorSpec` instance, to another space of type `target`.

    If the cast function is not found, `NotImplemented` is returned.
    """

    input_type = type(space)
    cast = _CASTERS[input_type, target_type]
    target_tspec, module = cast(space)
    return CastedModule(module, space, target_tspec)


def _is_tensor_spec_type(obj) -> typing.TypeIs[type[tspecs.TensorSpec]]:
    return isinstance(obj, type) and issubclass(obj, tspecs.TensorSpec)
