# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from collections import abc as cabc

from torch import nn


from aioway.tspecs import TSpec, TSpecInfer

__all__ = ["Instr"]


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
