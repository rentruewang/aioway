# Copyright (c) AIoWay Authors - All Rights Reserved

"A package of module and optimizer."

import dataclasses as dcls
import functools

import torch
from torch import nn, optim

from aioway._pytrees import tree_leaves
from aioway._utils import AnySet, any_set

__all__ = ["Mopt"]


@dcls.dataclass(frozen=True)
class Mopt:
    """
    A `Mopt` binds an `nn.Module` and `optim.Optimizer` together.
    """

    mod: nn.Module

    opt: optim.Optimizer

    def __post_init__(self) -> None:
        _ = self.parameters()

    def parameters(self):
        return self._parameters

    @functools.cached_property
    def _parameters(self) -> AnySet[torch.Tensor]:
        aset: AnySet[torch.Tensor] = any_set()
        for param in self.mod.parameters():
            aset.add(param)

        tensors = [
            t for t in tree_leaves(self.opt.param_groups) if isinstance(t, torch.Tensor)
        ]

        if aset != tensors:
            raise ValueError("The optimizer is not built for the module.")

        return aset
