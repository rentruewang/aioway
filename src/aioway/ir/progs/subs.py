# Copyright (c) AIoWay Authors - All Rights Reserved

import torch
from torch import nn
from torch.utils import _pytree as pyt

from aioway.ir.instrs import FuncCall, ModuleCall
from aioway.t import is_fake_tensor, parse_attr

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

    def hashes_of_modules(self) -> list[dict[str, int]]:
        state_dicts = [module.state_dict() for module in self._module_list]
        return pyt.tree_map_only(torch.Tensor, func=_hash_tensor, tree=state_dicts)


def _hash_tensor(tensor: torch.Tensor) -> int:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError

    if is_fake_tensor(tensor):
        return hash(parse_attr(tensor))

    else:
        return id(tensor)
