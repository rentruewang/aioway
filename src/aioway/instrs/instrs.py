# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Instr` interface."

import abc
import typing
from collections import abc as cabc

from torch import nn

from aioway.tspecs import TSpecInfer

__all__ = ["Instr"]

_NN_INSTR_REGISTRY: dict[type[nn.Module], type[Instr]] = {}
"The registry for `Instr`."

_NOT_CONCRETE_INSTR = nn.Module
"The marker that `Instr` is not a concrete class."


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

    NN: typing.ClassVar[type[nn.Module]] = _NOT_CONCRETE_INSTR
    """
    The `nn.Module` type that this `Instr` handles.
    """

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
    def lift(cls, module: nn.Module, /) -> typing.Self:
        """
        Converting from a concrete `nn.Module` into an `Instr`.
        """

        if not isinstance(module, cls.NN):
            raise TypeError(f"{cls} only handles {cls.NN}, but {type(module)=}.")

        return cls._lift(module)

    @classmethod
    @abc.abstractmethod
    def _lift(cls, module: nn.Module) -> typing.Self:
        raise NotImplementedError
