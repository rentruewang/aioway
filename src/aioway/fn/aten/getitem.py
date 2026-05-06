# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from collections import abc as cabc

import torch
from torch import ops

from aioway._common.dcls import dcls_frozen_no_repr

from ..fate import Fate

__all__ = ["BooleanMasking", "IntSelect"]


@dcls_frozen_no_repr
class _GetItem(Fate, abc.ABC):
    IR = ops.aten.index.Tensor

    self: torch.Tensor
    indices: list[torch.Tensor]

    def __hash__(self) -> int:
        return id(self)

    @typing.final
    def tensors(self) -> cabc.Iterator[torch.Tensor]:
        yield self.self
        yield from self.indices

    @typing.override
    def cost(self) -> int:
        return self.do().numel()


@dcls_frozen_no_repr
class BooleanMasking(_GetItem):
    def ok(self) -> bool:
        return len(self.indices) == 1 and self.indices[0].dtype == torch.bool

    @typing.override
    def do(self) -> torch.Tensor:
        return self.self


@dcls_frozen_no_repr
class IntSelect(_GetItem):
    def ok(self):
        return len(self.indices) == 1 and self.indices[0].dtype == torch.int

    @typing.override
    def do(self):
        return self.self[self.indices]
