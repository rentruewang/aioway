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
        return self.rewrite(module)

    @abc.abstractmethod
    def rewrite(self, module: I, /) -> O:
        raise NotImplementedError
