# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn

from aioway.ir.instrs import FuncCall, ModuleCall

from .progs import Program

__all__ = ["FuncProgram", "NnProgram"]


class FuncProgram(Program[FuncCall]):
    INSTR = FuncCall


class NnProgram(Program[ModuleCall]):
    INSTR = ModuleCall


class NnProgramModule(nn.Module):
    def __init__(self, program: NnProgram) -> None:
        super().__init__()

        self._program = program

        # Register the modules in `NnProgram` into a module.
        self._module_list = nn.ModuleList(instr.func for instr in self._program)

    def __len__(self) -> int:
        return len(self._module_list)

    def __getitem__(self, idx: int) -> nn.Module:
        return self._module_list[idx]

    @property
    def program(self) -> NnProgram:
        return self._program
