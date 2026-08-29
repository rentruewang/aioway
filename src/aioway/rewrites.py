# Copyright (c) AIoWay Authors - All Rights Reserved

"The rewriter module."

import typing

from aioway.instrs import Instr

__all__ = ["Rewriter"]


class Rewriter(typing.Protocol):
    """
    The rewriter rewrites an `Instr` into another.

    Note: Switch this to `nn.Module` -> `nn.Module`.
    """

    def __call__(self, instr: Instr) -> Instr: ...
