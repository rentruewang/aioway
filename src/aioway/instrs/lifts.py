# Copyright (c) AIoWay Authors - All Rights Reserved

"`Lift` parses existing `nn.Module` into `Instr`."

import dataclasses as dcls
import typing

from torch import nn

from .instrs import Instr

__all__ = ["lift", "Lift"]

_LIFTS: dict[type[nn.Module], Lift] = {}
"""
The registry storing all the `Lift` that are instantiated, by `nn.Module` type.
"""


class LiftFunc[M: nn.Module, I: Instr](typing.Protocol):
    def __call__(self, module: M, /) -> I: ...


class LiftFuncDecor[M: nn.Module, I: Instr](typing.Protocol):
    def __call__(self, func: LiftFunc[M, I], /) -> LiftFunc[M, I]: ...


def lift(module: nn.Module) -> Instr:
    module_type = type(module)
    rule = _LIFTS[module_type]
    return rule(module)


@typing.final
@dcls.dataclass(frozen=True)
class Lift[M: nn.Module = typing.Any, I: Instr = typing.Any]:
    """
    The class that is responsible for parsing `nn.Module` into `Instr`.
    """

    nn_type: type[M]
    """
    The source `nn.Module` type.
    """

    instr_type: type[I]
    """
    The target `Instr` type.
    """

    lift: LiftFunc[M, I]
    """
    The function that will convert `nn.Module` into `Instr`.
    """

    def __post_init__(self) -> None:
        if self.nn_type in _LIFTS:
            raise KeyError(self.nn_type)

        _LIFTS[self.nn_type] = self

    def __call__(self, module: nn.Module) -> Instr:
        if not isinstance(module, self.nn_type):
            raise TypeError(f"The module given is not of the type {self.nn_type}.")

        raise NotImplementedError

    @classmethod
    def register(
        cls, nn_type: type[nn.Module], instr_type: type[Instr]
    ) -> LiftFuncDecor:
        """
        Register a function that maps from `nn_type` to `instr_type`.
        """

        def decorator(lift: LiftFunc, /) -> LiftFunc:
            return Lift(nn_type=nn_type, instr_type=instr_type, lift=lift)

        return decorator
