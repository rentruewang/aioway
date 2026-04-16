# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

import torch
from torch import ops

from .fn import Preview


@dcls.dataclass(frozen=True)
class _BinaryTensorUFunc(Preview, abc.ABC):
    BINARY: cabc.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    self: torch.Tensor
    other: torch.Tensor
    alpha: float = 1

    @typing.override
    def ok(self) -> bool:
        try:
            _ = torch.broadcast_shapes(self.self.shape, self.other.shape)
            return True
        except RuntimeError:
            return False

    @typing.override
    def __call__(self) -> torch.Tensor:
        return self.BINARY(self.self, self.other * self.alpha)

    @typing.final
    @typing.override
    def tensors(self) -> cabc.Iterator[torch.Tensor]:
        yield self.self
        yield self.other

    @typing.final
    @typing.override
    def cost(self) -> int:
        return self._shape.numel()

    @functools.cached_property
    def _shape(self) -> torch.Size:
        return torch.broadcast_shapes(self.self.shape, self.other.shape)


class AddTensor(_BinaryTensorUFunc):
    IR = ops.aten.add.Tensor


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
