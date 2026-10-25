# Copyright (c) AIoWay Authors - All Rights Reserved

"A package of module and optimizer."

import dataclasses as dcls
import functools

import torch
from torch import nn, optim

from aioway.t import tree_leaves

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
    def _parameters(self) -> set[torch.Tensor]:
        sets: set[torch.Tensor] = {*self.mod.parameters()}

        tensors = {
            t for t in tree_leaves(self.opt.param_groups) if isinstance(t, torch.Tensor)
        }

        if sets != tensors:
            raise ValueError("The optimizer is not built for the module.")

        return sets
