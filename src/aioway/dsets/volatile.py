# Copyright (c) AIoWay Authors - All Rights Reserved

"The sources that are already in memory."

import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway._core import IndexibleIter, TensorIter, node_dcls

from .dsets import TdictFrame, dset_dcls

__all__ = ["TensorDictFrame", "TensorListIter", "TdictListIter"]


@typing.final
@dset_dcls
class TensorDictFrame(TdictFrame):
    """
    A `Frame` backed by a `td.TensorDict` (aka a batch in `aioway`).
    This means that it is non-distributed, and volatile.
    """

    data: td.TensorDict
    """
    The `td.TensorDict` source.
    """

    def _setup(self) -> None:
        self.data.auto_batch_size_()

    @typing.override
    def __len__(self) -> int:
        return len(self.data)

    @typing.override
    def __getitem__(self, idx: int) -> td.TensorDict:
        ret = self.data[idx]
        assert isinstance(ret, td.TensorDict)
        return ret

    @typing.override
    def __getitems__(self, idx: list[int]) -> td.TensorDict:
        ret = self.data[idx]
        assert isinstance(ret, td.TensorDict)
        return ret.auto_batch_size_()


@node_dcls
class TensorListIter(TensorIter):
    "A `Iter` backed by a list of `torch.Tensor`."

    sequence: cabc.Sequence[torch.Tensor]
    "List of `torch.Tensor`s."

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequence[idx]

    @property
    @typing.override
    def size(self) -> int:
        return len(self.sequence)

    def sample(self):
        return self.sequence[0]

    def iterate(self):
        for batch in self.sequence:
            yield batch


@node_dcls
class TdictListIter(IndexibleIter):
    "A `Stream` backed by a list of `TensorDict`."

    sequence: cabc.Sequence[td.TensorDict]
    "List of `td.TensorDict`s."

    @typing.override
    def __len__(self) -> int:
        return self.size

    @typing.override
    def __getitem__(self, idx: int) -> td.TensorDict:
        return self.sequence[idx]

    @property
    @typing.override
    def size(self) -> int:
        return len(self.sequence)

    def sample(self):
        return self.sequence[0]

    def iterate(self):
        for batch in self.sequence:
            yield batch
