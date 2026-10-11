# Copyright (c) AIoWay Authors - All Rights Reserved

"A unified interface for reducing a program to something else."

import abc
import typing

import loguru as L
import torch
from torch.utils import _pytree as pyt

from aioway.ir.instrs import Instr, InstrList
from aioway.ir.progs import Program
from aioway.t import TList, parse_attr

__all__ = ["Intrptr"]


class Intrptr[I: Instr = Instr, O: object = typing.Any](abc.ABC):
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
    def __call__(self, *args: typing.Any, **kwargs: typing.Any) -> O:
        # Find the tensors to bind.
        flattened = pyt.arg_tree_leaves(*args, **kwargs)
        tensors = [t for t in flattened if isinstance(t, torch.Tensor)]

        program = self.program

        if len(tensors) != len(program.inputs):
            raise TypeError(
                f"Cannot bind {len(tensors)} to {len(program.inputs)} input tensors."
            )

        if program.inputs.attrs() != [parse_attr(t) for t in tensors]:
            raise TypeError("Tensors do not look like they can be consumed by inputs.")

        L.logger.debug("Binding {l} tensors to respective inputs.", l=len(tensors))
        self.bind(program.inputs, tensors)

        L.logger.debug("Prepare walking the {} instructions.", len(program.instrs))
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
    def bind(self, inputs: TList, args: list[torch.Tensor], /) -> None:
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
    def finalize(self) -> O:
        """
        Finalize and return the result.
        """

        raise NotImplementedError
