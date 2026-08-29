# Copyright (c) AIoWay Authors - All Rights Reserved

"The rewriter module."
from aioway.instrs import Instr

import typing


class Rewriter(typing.Protocol):
    """
    The rewriter rewrites an `Instr` into another.
    """

    def __call__(self, instr: Instr) -> Instr: ...
