# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Instr` interface."

import abc
import inspect
import logging
import typing
from collections import abc as cabc

from torch import nn

if typing.TYPE_CHECKING:
    from .deducts import Deductor

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
    `Instr` is the IR `aioway` emits.

    It bridges several functionalities:

    - It emits `nn.Module`.
    - It transforms `TSpec` to match the underlying data change.
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
        if not cls.implements_nn():
            LOGGER.debug("%s is an abstract class.", cls)
            return

        if inspect.isabstract(cls):
            raise RuntimeError(
                f"{cls=} is abstract, but it should not be when {cls.NN=}."
            )

        LOGGER.debug("%s is registered into registry.", cls)
        _INSTRS_BY_MODULE[cls.NN] = cls

    @abc.abstractmethod
    def module(self) -> nn.Module:
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

    @classmethod
    def deductor(cls) -> Deductor:
        """
        Infer how an input described by `spec` would be converted to
        another item described by the output `TSpec`.

        The signature would match the `nn.Module.forward` signature,
        for example, `nn.Linear` would have a `Deductor` of signature `(input: tspecs.Unbounded)`.

        Returns:
            A `Deductor` that transforms the input argumnts `TSpec` to an output `TSpec`.
        """

        from .deducts import deductor_for

        return deductor_for(cls)

    @classmethod
    def deductor_is_defined(cls) -> bool:
        """
        Check if `cls.deductor()` returns a `Deductor` or not.

        This is useful in testing.
        """

        return len(cls.deductor()) != 0

    @classmethod
    def implements_nn(cls) -> bool:
        return _nn_is_defined(cls.NN)


class AiowayModule[I = typing.Any, O = typing.Any](nn.Module, abc.ABC):

    @abc.abstractmethod
    def forward(self, input: I, /) -> O:
        raise NotImplementedError


def _nn_is_defined(cls: type[nn.Module]) -> bool:
    return cls is not _NOT_CONCRETE_INSTR_NN
