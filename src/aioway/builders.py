# Copyright (c) AIoWay Authors - All Rights Reserved

"The builder for `Hop`s."

import dataclasses as dcls
import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway.dsets import FaissIndex, FaissIndexHop
from aioway.hop import (
    ApplyHop,
    BoundedHop,
    ColumnViewHop,
    FuncFilterHop,
    Hop,
    ListHop,
    NestedLoopJoinHop,
    NnLayerHop,
    NnLossHop,
    ProjectHop,
    RenameHop,
    StackHop,
    TdictHop,
    TensorHop,
    ZipHop,
)
from aioway.nn import BaseLoss, NnInit

__all__ = ["Builder"]


@typing.dataclass_transform(frozen_default=True)
def builder_dcls(cls):
    return dcls.dataclass(frozen=True)(cls)


@builder_dcls
class Builder:
    hop: Hop

    @classmethod
    def from_list(cls, items: cabc.Sequence[typing.Self]):
        hops = [item.hop for item in items]
        return cls(ListHop(hops))


@builder_dcls
class TensorBuilder(Builder):
    hop: TensorHop

    def create_index(self, query: TensorBuilder, k: int) -> typing.Self:
        index = FaissIndex.from_hop(self.hop)
        source = torch.cat(list(self.hop))
        return type(self)(FaissIndexHop(index, source=source, query=query.hop, k=k))

    def apply_layer(self, nn_init: NnInit) -> typing.Self:
        return type(self)(NnLayerHop(nn_init, module=nn_init.init_nn(), input=self.hop))

    def apply_loss(self, target: TensorBuilder, nn_init: NnInit) -> typing.Self:
        assert isinstance(nn_init, BaseLoss), type(nn_init)
        return type(self)(
            NnLossHop(nn_init, nn_init.init_nn(), input=self.hop, target=target.hop)
        )

    @classmethod
    def cat(cls, items: cabc.Sequence[TensorHop], dim: int = 0) -> typing.Self:
        return cls(StackHop(list(items), dim=dim))

    @classmethod
    def stack(cls, items: cabc.Sequence[TensorHop], dim: int = 0) -> typing.Self:
        return cls(StackHop(list(items), dim=dim))


@builder_dcls
class TdictBuilder(Builder):
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
