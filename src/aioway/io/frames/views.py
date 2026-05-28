# Copyright (c) AIoWay Authors - All Rights Reserved

"`View`s are columns and projections of tables."

import dataclasses as dcls
import typing

import tensordict as td
import torch

from aioway.dsets import DatasetColumnView, DatasetSelectView

from .frames import FrameDict

__all__ = ["FrameColumnView", "FrameSelectView"]


@dcls.dataclass(frozen=True)
class FrameColumnView(DatasetColumnView[FrameDict]):
    """
    A column reference to a `Frame`.
    Performs `__getitem__` on a `Frame`, then select the column.
    """

    def __len__(self) -> int:
        return len(self.dset)

    def __getitem__(self, idx, /) -> torch.Tensor:
        batch = self.dset[idx]
        return batch[self.col]

    @classmethod
    def from_column(cls, dataset: FrameDict, /, column: str) -> typing.Self:
        return cls(dataset, column)


@dcls.dataclass(frozen=True)
class FrameSelectView(DatasetSelectView[FrameDict], FrameDict):
    """
    A selection view on the `Frame`.
    """

    COLUMN_TYPE = FrameColumnView

    @typing.override
    def __len__(self) -> int:
        return len(self.dset)

    @typing.override
    def _getitems_batch(self, idx: list[int], /) -> td.TensorDict:
        items = self.dset[idx]
        return items.select(*self.cols)

    @classmethod
    def from_columns(cls, dataset: FrameDict, /, *columns: str) -> typing.Self:
        return cls(dataset, columns)
