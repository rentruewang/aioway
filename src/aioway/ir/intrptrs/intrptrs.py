# Copyright (c) AIoWay Authors - All Rights Reserved

"A unified interface for reducing a program to something else."

import abc
import typing

import loguru as L

from aioway.ir.instrs import Instr, InstrList
from aioway.ir.progs import Program
from aioway.t import TList

__all__ = ["Intrptr"]


class Intrptr[I: Instr = Instr, R = typing.Any](abc.ABC):
    """
    The `Intrptr` API interprets the program and process it to something else.

    It has 3 main functions:

    - `setup` binds the program's inputs to real input.
    - `walk` goes over the instruction list.
    - `finalize` yields the result.

    Right now it only binds `torch.Tensor` in the input,
    we may need more in the future to handle extra types.
    """

    @typing.final
    def __call__(self, *args, **kwargs) -> R:
        self.bind(self.program.inputs, *args, **kwargs)

        L.logger.debug("Prepare walking the {} instructions.", len(self.program.instrs))
        self.walk(self.program.instrs)

        L.logger.debug("Preparing the output.")
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
    def bind(self, inputs: TList, *args, **kwargs) -> None:
        """
        Setup with regards to the inputs and feed the data in.

        Only binds the tensors for now.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def walk(self, instrs: InstrList[I], /) -> None:
        """
        Walk over the program and update the states.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def finalize(self) -> R:
        """
        Finalize and return the result.
        """

        raise NotImplementedError
