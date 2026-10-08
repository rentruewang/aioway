# Copyright (c) AIoWay Authors - All Rights Reserved

"The rewriter module."

import abc
import typing

from aioway.ir import NnProgram

__all__ = ["Rewriter"]


class Rewriter(abc.ABC):
    """
    The rewriter rewrites an `Instr` into another.
    """

    @typing.no_type_check
    def __call__(self, module: NnProgram) -> NnProgram:
        if not self.handle(module):
            return NotImplemented

        return self.rewrite(module)

    @abc.abstractmethod
    def handle(self, module: NnProgram, /) -> bool:
        """
        Check whether the `Rewriter` handles the module or not.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def rewrite(self, module: typing.Any, /) -> NnProgram:
        """
        Perform the rewrite. Should not modify the input module.

        This shall not raise an exception, as it assumes input is valid.
        """

        raise NotImplementedError
