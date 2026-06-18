# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing

import torch
from torch import ops

from aioway._costs import Cost

from .fate import Aten, aten_dcls

__all__ = ["BooleanMasking", "IntSelect"]


@aten_dcls
class _BaseGetItem(Aten, abc.ABC):
    self: torch.Tensor
    indices: list[torch.Tensor]

    def __hash__(self) -> int:
        return id(self)

    @typing.override
    def cost(self) -> Cost:
        return Cost(time=self.self.numel(), memory=0)


@aten_dcls
class BooleanMasking(_BaseGetItem):
    KEY = ops.aten.index.Tensor

    def ok(self) -> bool:
        return len(self.indices) == 1 and self.indices[0].dtype == torch.bool

    @typing.override
    def forward(self) -> torch.Tensor:
        return self.self


@aten_dcls
class IntSelect(_BaseGetItem):
    KEY = ops.aten.index.Tensor

    def ok(self):
        return len(self.indices) == 1 and self.indices[0].dtype == torch.int
