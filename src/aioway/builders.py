# Copyright (c) AIoWay Authors - All Rights Reserved

"The builder for `Iter`s."

import dataclasses as dcls
import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway._core import BoundedIter, Iter, ListIter, TdictIter, TensorIter
from aioway.hop import (
    ApplyIter,
    ColumnViewIter,
    FuncFilterIter,
    NestedLoopJoinIter,
    ProjectIter,
    RenameIter,
    StackIter,
    ZipIter,
)
from aioway.indices import AnnIndexTrainerIter, FaissIndex
from aioway.torch.nn import BaseLoss, NnInit, NnLayerIter, NnLossIter

__all__ = ["Builder"]


@typing.dataclass_transform(frozen_default=True)
def builder_dcls(cls):
    return dcls.dataclass(frozen=True)(cls)


@builder_dcls
class Builder:
    hop: Iter

    @classmethod
    def from_list(cls, items: cabc.Sequence[typing.Self]):
        hops = [item.hop for item in items]
        return cls(ListIter(hops))


@builder_dcls
class TensorBuilder(Builder):
    hop: TensorIter

    def create_ann_index(self) -> Builder:
        "Create an ANN index builder that trains an index when evaluated."

        shape = self.hop.attr.shape

        if shape.ndim != 2:
            raise RuntimeError(f"Since {shape} is not 2D, cannot create ANN index.")

        index = FaissIndex(dim=self.hop.attr.shape[-1])

        trainer = AnnIndexTrainerIter(index, self.hop)
        return Builder(trainer)

    def apply_layer(self, nn_init: NnInit) -> typing.Self:
        return type(self)(
            NnLayerIter(nn_init, module=nn_init.init_nn(), input=self.hop)
        )

    def apply_loss(self, target: TensorBuilder, nn_init: NnInit) -> typing.Self:
        assert isinstance(nn_init, BaseLoss), type(nn_init)
        return type(self)(
            NnLossIter(nn_init, nn_init.init_nn(), input=self.hop, target=target.hop)
        )

    @classmethod
    def cat(cls, items: cabc.Sequence[TensorIter], dim: int = 0) -> typing.Self:
        return cls(StackIter(list(items), dim=dim))

    @classmethod
    def stack(cls, items: cabc.Sequence[TensorIter], dim: int = 0) -> typing.Self:
        return cls(StackIter(list(items), dim=dim))


@builder_dcls
class TdictBuilder(Builder):
    hop: TdictIter

    def column(self, col: str) -> TensorBuilder:
        return TensorBuilder(ColumnViewIter(self.hop, col))

    def select(self, *cols: str) -> typing.Self:
        return type(self)(ProjectIter(self.hop, list(cols)))

    def apply(self, func: cabc.Callable[[td.TensorDict], td.TensorDict]) -> typing.Self:
        return type(self)(ApplyIter(self.hop, func))

    def filter(self, func: cabc.Callable[[td.TensorDict], torch.Tensor]) -> typing.Self:
        return type(self)(FuncFilterIter(self.hop, func))

    def rename(self, **renames: str) -> typing.Self:
        return type(self)(RenameIter(self.hop, renames))

    def join(self, right: TdictBuilder, on: str) -> typing.Self:
        if not isinstance(right.hop, BoundedIter):
            raise TypeError(f"{right.hop=} must be bounded.")

        return type(self)(NestedLoopJoinIter(self.hop, right.hop, key=on))

    def zip(self, right: TdictBuilder) -> typing.Self:
        return type(self)(ZipIter(self.hop, right.hop))
