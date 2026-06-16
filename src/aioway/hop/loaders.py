# Copyright (c) AIoWay Authors - All Rights Reserved

"The interface for the `io` package."

import dataclasses as dcls
import typing
from collections import abc as cabc

import tensordict as td
import torch
from torch.utils import data

from aioway._utils import HasLen

from .hop import Hop, TdictHop, TensorHop, hop_dcls

__all__ = ["LoaderOpt", "LoaderHop", "TensorLoaderHop", "TdictLoaderHop"]


@dcls.dataclass(frozen=True)
class LoaderOpt:
    """
    The optoins for `data.DataLoader` on `Frame` in `FrameStream`.
    """

    batch_size: int = 1
    "The batch size of individual batches."

    drop_last: bool = False
    "Whether to drop the last batch, which may have a different `batch_size`."

    shuffle: bool = False
    "To shuffle or not."

    sampler: data.Sampler[int] | None = None
    "How to sample in case when want to shuffle."


@hop_dcls
class LoaderHop[T = typing.Any](Hop[T]):
    """
    A `Hop` backed by a `torch` `DataLoader`.
    """

    _: dcls.KW_ONLY

    dset: data.Dataset[T]
    """
    The data loader that would be iterated over.
    """

    opts: LoaderOpt = LoaderOpt()
    """
    The options to pass to `DataLoader`.
    """

    @property
    @typing.override
    def size(self) -> int:
        # May be a frame or stream, so we need to test.
        # Perhaps this is a bad design.
        if not isinstance(self.dset, HasLen):
            return NotImplemented

        # Drop last or not would affect counts.
        count, remain = divmod(len(self.dset), self.opts.batch_size)

        if remain and not self.opts.drop_last:
            count += 1

        return count

    def iterate(self) -> cabc.Generator[T]:
        yield from self._dataloader()

    def _dataloader(self) -> data.DataLoader[T]:
        return data.DataLoader(self.dset, **dcls.asdict(self.opts))


@hop_dcls
class TensorLoaderHop(LoaderHop[torch.Tensor], TensorHop):
    """
    A `Hop` to load `torch.Tensor`.
    """


@hop_dcls
class TdictLoaderHop(LoaderHop[td.TensorDict], TdictHop):
    """
    A `Hop` to load `td.TensorDict`.
    """

    @typing.override
    def _dataloader(self) -> data.DataLoader[td.TensorDict]:
        return data.DataLoader(self.dset, collate_fn=td.stack, **dcls.asdict(self.opts))
