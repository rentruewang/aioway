# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing

import torch
from torch import ops

from aioway._types import dcls_no_repr

from .fate import Fate

__all__ = ["BooleanMasking", "IntSelect"]


@dcls_no_repr
class _GetItem(Fate, abc.ABC):
    self: torch.Tensor
    indices: list[torch.Tensor]

    if typing.TYPE_CHECKING:
        # GetItem has `torch.Tensor` as output.
        def do(self) -> torch.Tensor: ...

    def __hash__(self) -> int:
        return id(self)

    @typing.override
    def cost(self) -> int:
        return self.do().numel()


@dcls_no_repr
class BooleanMasking(_GetItem, key=ops.aten.index.Tensor):

    def ok(self) -> bool:
        return len(self.indices) == 1 and self.indices[0].dtype == torch.bool

    @typing.override
    def do(self) -> torch.Tensor:
        return self.self


@dcls_no_repr
class IntSelect(_GetItem, key=ops.aten.index.Tensor):

    def ok(self):
        return len(self.indices) == 1 and self.indices[0].dtype == torch.int
