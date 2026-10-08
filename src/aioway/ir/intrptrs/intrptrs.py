# Copyright (c) AIoWay Authors - All Rights Reserved

"A unified interface for reducing a program to something else."

import abc
import typing

import torch

from aioway.ir.instrs import Instr, InstrList
from aioway.ir.progs import Program
from aioway.t import TList

__all__ = ["Intrptr"]


class Intrptr[I: Instr = Instr, T: object = typing.Any](abc.ABC):
    """
    The `Intrptr` API interprets the program and process it to something else.

    It has 3 main functions:

    - `setup` binds the program's inputs to real input.
    - `walk` goes over the instruction list.
    - `finalize` yields the result.

    Right now it only allows `torch.Tensor` list as input.
    Consider generalize it in the future.
    """

    @typing.final
    def __call__(self, *inputs: torch.Tensor) -> T:
        self.setup(self.program.inputs, *inputs)
        self.walk(self.program.instrs)
        return self.finalize()

    @property
    @abc.abstractmethod
    def program(self) -> Program[I]:
        """
        The property that signals that `Intrptr` is bound to a `Program`.
        Exists for type safety.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def setup(self, inputs: TList, *data: torch.Tensor) -> None:
        """
        Setup with regards to the inputs and feed the data in.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def walk(self, instrs: InstrList[I], /) -> None:
        """
        Walk over the program and update the states.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def finalize(self) -> T:
        """
        Finalize and return the result.
        """

        raise NotImplementedError
