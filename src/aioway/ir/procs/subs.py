# Copyright (c) AIoWay Authors - All Rights Reserved

import torch
from torch import nn

from aioway.ir.instrs import FuncCall, ModuleCall
from aioway.t import is_fake_tensor, parse_attr, tree_map_only

from .procs import Proc

__all__ = ["FuncProc", "NnProc", "NnProcModule"]


class FuncProc(Proc[FuncCall]):
    INSTR = FuncCall


class NnProc(Proc[ModuleCall]):
    INSTR = ModuleCall

    @property
    def module(self) -> NnProcModule:
        return NnProcModule(self)


class NnProcModule(nn.Module):
    def __init__(self, program: NnProc) -> None:
        super().__init__()

        self._proc = program

        # Register the modules in `NnProc` into a module.
        self._module_list = nn.ModuleList(instr.func for instr in self._proc)

    def __len__(self) -> int:
        return len(self._module_list)

    def __getitem__(self, idx: int) -> nn.Module:
        return self._module_list[idx]

    def forward(self, *args, **kwargs):
        from aioway.ir import Exec

        executor = Exec(self._proc)
        return executor(*args, **kwargs)

    @property
    def proc(self) -> NnProc:
        return self._proc

    def hashes_of_modules(self) -> list[dict[str, int]]:
        state_dicts = [module.state_dict() for module in self._module_list]
        return tree_map_only(torch.Tensor, func=_hash_tensor, tree=state_dicts)


def _hash_tensor(tensor: torch.Tensor) -> int:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError

    if is_fake_tensor(tensor):
        return hash(parse_attr(tensor))

    else:
        return id(tensor)
