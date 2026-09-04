# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn

from aioway.tspecs import TSpec

from .instrs import Instr
from .nn import NnInstr, instr_dcls

__all__ = ["Sequential"]


@instr_dcls
class Sequential(NnInstr):
    """
    The wrapper for `nn.Sequential`.

    Since the API of `nn.Sequential` takes in `*nn.Module`s,
    and there is no easy way of making that into a dataclass,
    we must overwrite `.__call__` because the default is `NN(**dcls.asdict(self))`.
    """

    NN = nn.Sequential

    modules: tuple[Instr, ...]
    """
    A list of already initialized `nn.Module` objects.
    """

    def __init__(self, *args: Instr):
        super().__init__()
        self.modules = args

    @typing.override
    def module(self) -> nn.Sequential:
        # Create `nn.Sequential` instance with `NnInitFn` is the best way
        # to ensure that the modes are invoked properly.
        modules = [mo.module() for mo in self.modules]
        return self.NN(*modules)

    @typing.override
    def children(self):
        yield from self.modules


@Sequential.deducer().register
def sequential_deduct(self: Sequential, input: TSpec) -> TSpec:
    for sub in self.modules:
        deducer = sub.deducer()

        if (output := deducer(sub, input)) is NotImplemented:
            return NotImplemented

        input = output
    return input
