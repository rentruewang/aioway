# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from torch import ops

from ..fn import Preview


@dcls.dataclass(frozen=True)
class _GetItem(Preview, abc.ABC):
    IR = ops.aten.index.Tensor

    self: torch.Tensor
    indices: list[torch.Tensor]

    @typing.final
    def tensors(self) -> cabc.Iterator[torch.Tensor]:
        yield self.self
        yield from self.indices


@dcls.dataclass(frozen=True)
class BooleanMasking(_GetItem):
    def ok(self) -> bool:
        return len(self.indices) == 1 and self.indices[0].dtype == torch.bool

    @typing.override
    def __call__(self) -> torch.Tensor:
        return self.self

    @typing.override
    def cost(self) -> int:
        return self().numel()


@dcls.dataclass(frozen=True)
class IntSelect(_GetItem):
    def ok(self):
        return len(self.indices) == 1 and self.indices[0].dtype == torch.int

    @typing.override
    def __call__(self):
        return self.self[self.indices]

    @typing.override
    def cost(self) -> int:
        return self().numel()
