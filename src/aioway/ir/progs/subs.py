# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway.ir.instrs import FuncCall, ModuleCall

from .progs import Program


class FuncProgram(Program[FuncCall]):
    pass


class NnProgram(Program[ModuleCall]):
    pass
