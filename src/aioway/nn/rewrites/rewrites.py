# Copyright (c) AIoWay Authors - All Rights Reserved

"The rewriter module."

import typing

from torch import nn

__all__ = ["Rewriter"]


class Rewriter(typing.Protocol):
    """
    The rewriter rewrites an `Instr` into another.
    """

    def __call__(self, module: nn.Module, /) -> nn.Module: ...
