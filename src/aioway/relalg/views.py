# Copyright (c) AIoWay Authors - All Rights Reserved

"Columns of `Exec`."

import dataclasses as dcls
import typing
from collections import abc as cabc

import tensordict as td
import torch

from .execs import TdictExec, TensorExec, node_dcls
from .maps import MapExec

__all__ = ["ColumnViewExec", "ProjectExec"]


@node_dcls
class ColumnViewExec(TensorExec):
    """
    A column reference (on a `Exec`).
    Performs `__next__` and yield `torch.Tensor`s.
    """

    input: TdictExec
    "The input `TdictExec` to perform views on."

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
class ProjectExec(MapExec):
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
