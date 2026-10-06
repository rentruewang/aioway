# Copyright (c) AIoWay Authors - All Rights Reserved

"The `nn.Module` version of `Program`."

import typing
from collections import abc as cabc

import torch

from aioway.ir.instrs import InstrList, ModuleCall

from .progs import Program

__all__ = ["NnProgram"]


class NnProgram(Program[ModuleCall]):
    def __init__(
        self,
        instrs: cabc.Iterable[ModuleCall],
        inputs: cabc.Iterable[torch.Tensor],
        outputs: cabc.Iterable[torch.Tensor],
    ) -> None:
        super().__init__(instrs, inputs, outputs)

        for instr in instrs:
            if not isinstance(instr, ModuleCall):
                raise TypeError("`NnProgram` only accepts `ModuleCall`.")

    @property
    def instrs(self) -> InstrList:
        result: typing.Any = self._instrs
        return result
