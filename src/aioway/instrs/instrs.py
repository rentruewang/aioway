# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Instr` interface."

import abc
import inspect
import logging
import typing
from collections import abc as cabc

from torch import nn

from .deducts import TSpecInfer

__all__ = ["Instr", "AiowayModule"]

LOGGER = logging.getLogger(__name__)

_NN_INSTR_REGISTRY: dict[type[nn.Module], type[Instr]] = {}
"The registry for `Instr`."

_NOT_CONCRETE_INSTR_NN = nn.Module
"The marker that `Instr` is not a concrete class."

_INSTRS_BY_MODULE: dict[type[nn.Module], type[Instr]] = {}
"""
The registry storing all the `Instr`s corresponding to their `nn.Module` type.
"""


class Instr[I = typing.Any, O = typing.Any](abc.ABC):
    """
    `NeInstrt` is the module type that `aioway` emits.

    It requires several methods to be implemented.

    `__tspec_infer__`: Describe how `forward` transforms data.
    """

    __match_args__: typing.ClassVar[tuple[str, ...]]
    """
    `Instr` should be able to be decomposed.
    """

    NN: typing.ClassVar[type[nn.Module]] = _NOT_CONCRETE_INSTR_NN
    """
    The `nn.Module` type that this `Instr` handles.
    """

    def __init_subclass__(cls) -> None:
        # Don't do anything if `cls.NN` is not updated.
        if _nn_not_defined(cls.NN):
            LOGGER.debug("%s is an abstract class.", cls)
            return

        if inspect.isabstract(cls):
            raise RuntimeError(
                f"{cls=} is abstract, but it should not be when {cls.NN=}."
            )

        LOGGER.debug("%s is registered into registry.", cls)
        _INSTRS_BY_MODULE[cls.NN] = cls

    @abc.abstractmethod
    def __tspec_infer__(self) -> TSpecInfer:
        """
        Infer how an input described by `spec` would be converted to
        another item described by the output `TSpec`.

        Args:
            spec: The input `TSpec`. This would `.contains` valid input to `forward`.

        Returns:
            A `TSpec` that describes the output to `forward`.
            Should return `NotImplemented` for non-supported input.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def module(self) -> AiowayModule:
        """
        Build the module represented by this current `Instr`.
        This function is responsible for recursively construct sub-modules as well,
        that are emitted by the `Instr` represented by children.

        This transparently leverages the `fake_mode`, if we are running under it,
        then the `nn.Module` would be trivial to construct. In some cases,
        the `Instr` may contain already initailized `nn.Module`, and returning it is allowed.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def children(self) -> cabc.Iterable[Instr]:
        """
        Get the children of this current class.
        """

        raise NotImplementedError


class AiowayModule[I = typing.Any, O = typing.Any](nn.Module, abc.ABC):

    @abc.abstractmethod
    def forward(self, input: I, /) -> O:
        raise NotImplementedError


def _nn_not_defined(cls: type[nn.Module]) -> bool:
    return cls is _NOT_CONCRETE_INSTR_NN
