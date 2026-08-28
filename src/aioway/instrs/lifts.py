# Copyright (c) AIoWay Authors - All Rights Reserved

"`Lift` parses existing `nn.Module` into `Instr`."

import dataclasses as dcls
import typing

from torch import nn

from .instrs import Instr

__all__ = ["lift", "LiftRule", "list_lift_rules"]

_LIFT_RULES: dict[type[nn.Module], LiftRule] = {}
"""
The registry storing all the `LiftRule`s that are instantiated, by `nn.Module` type.
"""


class LiftFunc[M: nn.Module, I: Instr](typing.Protocol):
    def __call__(self, module: M, /) -> I: ...


class LiftFuncDecor[M: nn.Module, I: Instr](typing.Protocol):
    def __call__(self, func: LiftFunc[M, I], /) -> LiftFunc[M, I]: ...


def lift(module: nn.Module) -> Instr:
    """
    A function that converts the `nn.Module` given to `Instr`,
    based on the rules registered in the global registry.
    """

    module_type = type(module)
    rule = _LIFT_RULES[module_type]
    return rule(module)


def list_lift_rules() -> list[LiftRule]:
    """
    List all the `LiftRule`s currently registered.
    """

    return list(_LIFT_RULES.values())


@typing.final
@dcls.dataclass(frozen=True)
class LiftRule[M: nn.Module = typing.Any, I: Instr = typing.Any]:
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
        if self.nn_type in _LIFT_RULES:
            raise KeyError(self.nn_type)

        _LIFT_RULES[self.nn_type] = self

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
            return LiftRule(nn_type=nn_type, instr_type=instr_type, lift=lift)

        return decorator
