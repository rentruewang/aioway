# Copyright (c) AIoWay Authors - All Rights Reserved

"The sources that are already in memory."

import typing

import tensordict as td

from aioway.attrs import AttrDict

from .dsets import TdictFrame, dset_dcls

__all__ = ["TensorDictFrame"]


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

    @typing.override
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> td.TensorDict:
        ret = self.data[idx]
        assert isinstance(ret, td.TensorDict)
        return ret

    @typing.override
    def __getitems__(self, idx: list[int]) -> td.TensorDict:
        ret = self.data[idx]
        assert isinstance(ret, td.TensorDict)
        return ret.auto_batch_size_()

    @property
    @typing.override
    def attrs(self) -> AttrDict:
        return AttrDict.parse(self.data)
