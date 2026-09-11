# Copyright (c) AIoWay Authors - All Rights Reserved

"The rewriter module."

import abc
import typing

from torch import nn

__all__ = ["Rewriter"]


class Rewriter[I: nn.Module = typing.Any, O: nn.Module = typing.Any](abc.ABC):
    """
    The rewriter rewrites an `Instr` into another.
    """

    @typing.no_type_check
    def __call__(self, module: nn.Module) -> nn.Module:
        if not self.handle(module):
            return NotImplemented

        return self.rewrite(module)

    @abc.abstractmethod
    def handle(self, module: I, /) -> bool:
        """
        Check whether the `Rewriter` handles the module or not.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def rewrite(self, module: I, /) -> O:
        """
        Perform the rewrite. Should not modify the input module.

        This shall not raise an exception, as it assumes input is valid.
        """

        raise NotImplementedError
