# Copyright (c) AIoWay Authors - All Rights Reserved

"`HopColumn`s are a column of `Hop`."

import dataclasses as dcls
import typing
from collections import abc as cabc

import tensordict as td
import torch

from .hop import TdictHop, TensorHop, hop_dcls
from .maps import MapHop

__all__ = ["ColumnViewHop", "ProjectHop"]


@hop_dcls
class ColumnViewHop(TensorHop):
    """
    A column reference (on a `Hop`).
    Performs `__next__` and yield `torch.Tensor`s.
    """

    input: TdictHop
    "The input `TdictHop` to perform views on."

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


@hop_dcls
class ProjectHop(MapHop):
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
