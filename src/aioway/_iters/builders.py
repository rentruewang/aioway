# Copyright (c) AIoWay Authors - All Rights Reserved

"The builder for `Iter`s."

import dataclasses as dcls
import typing
from collections import abc as cabc

import tensordict as td
import torch
from torch import nn

from aioway._iters import IndexibleIter, Iter, ListIter, TdictIter, TensorIter
from aioway.indices import AnnIndexTrainerIter, FaissIndex
from aioway.relalg import (
    ApplyIter,
    ColumnViewIter,
    FuncFilterIter,
    NestedLoopJoinIter,
    ProjectIter,
    RenameIter,
    StackIter,
    ZipIter,
)
from aioway.torch.nn import NnLayerUFunc, NnLossUFunc

__all__ = ["Builder"]


@typing.dataclass_transform(frozen_default=True)
def builder_dcls(cls):
    return dcls.dataclass(frozen=True)(cls)


@builder_dcls
class Builder:
    iterator: Iter

    @classmethod
    def from_list(cls, items: cabc.Sequence[typing.Self]):
        hops = [item.iterator for item in items]
        return cls(ListIter(hops))


@builder_dcls
class TensorBuilder(Builder):
    iterator: TensorIter

    def create_ann_index(self) -> Builder:
        "Create an ANN index builder that trains an index when evaluated."

        shape = self.iterator.attr.shape

        if shape.ndim != 2:
            raise RuntimeError(f"Since {shape} is not 2D, cannot create ANN index.")

        index = FaissIndex(dim=self.iterator.attr.shape[-1])

        trainer = AnnIndexTrainerIter(index, self.iterator)
        return Builder(trainer)

    def apply_layer[**P, T: nn.Module](
        self, module: cabc.Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Builder:
        ufunc = NnLayerUFunc(module, *args, **kwargs)
        return Builder(ufunc.thunk(self.iterator))

    def apply_loss[**P, T: nn.Module](
        self,
        target: TensorBuilder,
        module: cabc.Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Builder:
        ufunc = NnLossUFunc(module, *args, **kwargs)
        return Builder(ufunc.thunk(self.iterator, target.iterator))

    @classmethod
    def cat(cls, items: cabc.Sequence[TensorIter], dim: int = 0) -> typing.Self:
        return cls(StackIter(list(items), dim=dim))

    @classmethod
    def stack(cls, items: cabc.Sequence[TensorIter], dim: int = 0) -> typing.Self:
        return cls(StackIter(list(items), dim=dim))


@builder_dcls
class TdictBuilder(Builder):
    iterator: TdictIter

    def column(self, col: str) -> TensorBuilder:
        return TensorBuilder(ColumnViewIter(self.iterator, col))

    def select(self, *cols: str) -> typing.Self:
        return type(self)(ProjectIter(self.iterator, list(cols)))

    def apply(self, func: cabc.Callable[[td.TensorDict], td.TensorDict]) -> typing.Self:
        return type(self)(ApplyIter(self.iterator, func))

    def filter(self, func: cabc.Callable[[td.TensorDict], torch.Tensor]) -> typing.Self:
        return type(self)(FuncFilterIter(self.iterator, func))

    def rename(self, **renames: str) -> typing.Self:
        return type(self)(RenameIter(self.iterator, renames))

    def join(self, right: TdictBuilder, on: str) -> typing.Self:
        if not isinstance(right.iterator, IndexibleIter):
            raise TypeError(f"{right.iterator=} must be bounded.")

        return type(self)(NestedLoopJoinIter(self.iterator, right.iterator, key=on))

    def zip(self, right: TdictBuilder) -> typing.Self:
        return type(self)(ZipIter(self.iterator, right.iterator))
