# Copyright (c) AIoWay Authors - All Rights Reserved

"The sources that are already in memory."

import dataclasses as dcls
import typing

import tensordict as td

from .dsets import TdictFrame

__all__ = ["TensorDictFrame", "dset_dcls"]


@typing.dataclass_transform()
def dset_dcls(cls):
    return dcls.dataclass(cls)


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
