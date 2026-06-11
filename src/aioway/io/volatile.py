# Copyright (c) AIoWay Authors - All Rights Reserved

"The sources that are already in memory."

import functools
import typing
from collections import abc as cabc

import tensordict as td

from aioway.attrs import AttrDict
from aioway.hop import BoundedHop, TdictHop, hop_dcls

from .dsets import TdictFrame, dset_dcls

__all__ = ["TensorDictFrame", "SourceListHop"]


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

    def __post_init__(self) -> None:
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


@hop_dcls
class SourceListHop(BoundedHop):
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

    @functools.cached_property
    def _schema(self) -> AttrDict:
        schemas = [AttrDict.parse(chunk) for chunk in self.sequence]

        if len({*schemas}) == 1:
            return schemas[0]

        raise ValueError("Chunks should have the same schema.")

    def iterate(self):
        for batch in self.sequence:
            yield batch

    @classmethod
    def exhaust(cls, stream: TdictHop) -> typing.Self:
        return cls(list(stream))
