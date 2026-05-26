# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing

import torch
from torch import ops

from .fate import Fate, fate_dcls

__all__ = ["BooleanMasking", "IntSelect"]


@fate_dcls
class _BaseGetItem(Fate, abc.ABC):
    self: torch.Tensor
    indices: list[torch.Tensor]

    def __hash__(self) -> int:
        return id(self)

    @typing.override
    def cost(self) -> int:
        return self().numel()


@fate_dcls
class BooleanMasking(_BaseGetItem):
    KEY = ops.aten.index.Tensor

    def ok(self) -> bool:
        return len(self.indices) == 1 and self.indices[0].dtype == torch.bool

    @typing.override
    def forward(self) -> torch.Tensor:
        return self.self


@fate_dcls
class IntSelect(_BaseGetItem):
    KEY = ops.aten.index.Tensor

    def ok(self):
        return len(self.indices) == 1 and self.indices[0].dtype == torch.int
