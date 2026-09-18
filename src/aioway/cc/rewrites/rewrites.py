# Copyright (c) AIoWay Authors - All Rights Reserved

"The rewriter module."

import abc
import typing

from torch import nn

__all__ = ["Rewriter"]


class Rewriter(abc.ABC):
    """
    The rewriter rewrites an `Instr` into another.
    """

    @typing.no_type_check
    def __call__(self, module: nn.Module) -> nn.Module:
        if not self.handle(module):
            return NotImplemented

        return self.rewrite(module)

    @abc.abstractmethod
    def handle(self, module: nn.Module, /) -> bool:
        """
        Check whether the `Rewriter` handles the module or not.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def rewrite(self, module: typing.Any, /) -> nn.Module:
        """
        Perform the rewrite. Should not modify the input module.

        This shall not raise an exception, as it assumes input is valid.
        """

        raise NotImplementedError
