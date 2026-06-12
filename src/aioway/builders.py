# Copyright (c) AIoWay Authors - All Rights Reserved

"The builder for `Hop`s."

import dataclasses as dcls
import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway.hop import BoundedHop, Hop, StackHop, TdictHop, TensorHop
from aioway.io import FaissIndex, FaissIndexHop
from aioway.relalg import (
    ApplyHop,
    ColumnViewHop,
    FuncFilterHop,
    NestedLoopJoinHop,
    ProjectHop,
    RenameHop,
    ZipHop,
)

__all__ = ["Builder"]


@typing.dataclass_transform(frozen_default=True)
def builder_dcls(cls):
    return dcls.dataclass(frozen=True)(cls)


@builder_dcls
class Builder[H: Hop]:
    hop: H

    def unwrap(self) -> H:
        return self.hop


@builder_dcls
class TensorBuilder(Builder[TensorHop]):
    hop: TensorHop

    def create_index(self, query: TdictHop, k: int) -> typing.Self:
        index = FaissIndex.from_hop(self)
        source = torch.cat(list(self.hop))
        return type(self)(FaissIndexHop(index, source=source, query=query, k=k))

    @classmethod
    def cat(cls, items: cabc.Sequence[TensorHop], dim: int = 0) -> typing.Self:
        return cls(StackHop(list(items), dim=dim))

    @classmethod
    def stack(cls, items: cabc.Sequence[TensorHop], dim: int = 0) -> typing.Self:
        return cls(StackHop(list(items), dim=dim))


@builder_dcls
class TdictBuilder(Builder[TdictHop]):
    hop: TdictHop

    def column(self, col: str) -> TensorBuilder:
        return TensorBuilder(ColumnViewHop(self.hop, col))

    def select(self, *cols: str) -> typing.Self:
        return type(self)(ProjectHop(self.hop, list(cols)))

    def apply(self, func: cabc.Callable[[td.TensorDict], td.TensorDict]) -> typing.Self:
        return type(self)(ApplyHop(self.hop, func))

    def filter(self, func: cabc.Callable[[td.TensorDict], torch.Tensor]) -> typing.Self:
        return type(self)(FuncFilterHop(self.hop, func))

    def rename(self, **renames: str) -> typing.Self:
        return type(self)(RenameHop(self.hop, renames))

    def join(self, right: TdictBuilder, on: str) -> typing.Self:
        if not isinstance(right.hop, BoundedHop):
            raise TypeError(f"{right.hop=} must be bounded.")

        return type(self)(NestedLoopJoinHop(self.hop, right.hop, key=on))

    def zip(self, right: TdictBuilder) -> typing.Self:
        return type(self)(ZipHop(self.hop, right.hop))
