# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway.ir.instrs import FuncCall, ModuleCall

from .progs import Program

__all__ = ["FuncProgram", "NnProgram"]


class FuncProgram(Program[FuncCall]):
    INSTR = FuncCall


class NnProgram(Program[ModuleCall]):
    INSTR = ModuleCall
