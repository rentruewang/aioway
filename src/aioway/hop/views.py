# Copyright (c) AIoWay Authors - All Rights Reserved

"Columns of `Iter`."

import dataclasses as dcls
import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway._core import TdictIter, TensorIter, node_dcls

from .maps import MapIter

__all__ = ["ColumnViewIter", "ProjectIter"]


@node_dcls
class ColumnViewIter(TensorIter):
    """
    A column reference (on a `Iter`).
    Performs `__next__` and yield `torch.Tensor`s.
    """

    input: TdictIter
    "The input `TdictIter` to perform views on."

    col: str
    "The column to view on."

    @typing.override
    def iterate(self) -> cabc.Generator[torch.Tensor]:
        for batch in self.input:
            result = batch[self.col]
            assert isinstance(result, torch.Tensor)
            yield result

    @property
    @typing.override
    def size(self) -> int:
        return self.input.size


@node_dcls
class ProjectIter(MapIter):
    """
    Projection of the input table. The `subset` should be a subset of the input columns.
    """

    subset: list[str] = dcls.field(default_factory=list)
    """
    Input columns that appears in the outputs.
    """

    @typing.override
    def _apply(self, batch: td.TensorDict) -> td.TensorDict:
        return batch.select(*self.subset)
