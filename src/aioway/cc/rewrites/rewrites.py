# Copyright (c) AIoWay Authors - All Rights Reserved

"The rewriter module."

from aioway.ir import Query
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
    def query(self) -> Query:
        "The query that the rewriter gets."
        raise NotImplementedError

    @abc.abstractmethod
    def proposed(self) -> NnProgram:
        raise NotImplementedError

    @abc.abstractmethod
    def rewrite(self, module: typing.Any, /) -> NnProgram:
        """
        Perform the rewrite. Should not modify the input module.

        This shall not raise an exception, as it assumes input is valid.
        """

        raise NotImplementedError
