# Copyright (c) AIoWay Authors - All Rights Reserved

"`Frame`s that produce data by slicing contiguous input records."

import dataclasses as dcls
import functools
import typing

import numpy as np
import tensordict as td

from aioway._utils import IntArray, is_list_of
from aioway.attrs import AttrDict

from ..dsets import TdictFrame, dset_dcls

__all__ = ["TensorDictFrame", "TensorDictListFrame"]


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

    @typing.override
    def __getitems__(self, idx: list[int]) -> td.TensorDict:
        return self.data[idx].auto_batch_size_()

    @property
    @typing.override
    def attrs(self) -> AttrDict:
        return AttrDict.parse(self.data)


@typing.final
@dset_dcls
class TensorDictListFrame(TdictFrame):
    """
    A `Frame` backed by a `list[td.TensorDict]` (aka a batch in `aioway`).
    This means that it is non-distributed, and volatile.
    """

    _list: list[td.TensorDict] = dcls.field(default_factory=list)
    """
    The `list` of `td.TensorDict`s.
    The data must all have the same keys and data types (#100),
    but not necessarily the same `batch_size`.
    """

    def __post_init__(self) -> None:
        is_list_of_chunks = is_list_of(td.TensorDict)
        if not is_list_of_chunks(self._list):
            raise ValueError(f"Expected a list of `td.TensorDict`s. Got {self._list=}")

        # Check if `attrs` can be evaluated. If not, the tensordicts are malformed.
        _ = self.attrs

    @typing.override
    def __len__(self) -> int:
        return self._cumsum_len[-1]

    def append(self, td: td.TensorDict, /) -> None:
        self._list.append(td)

    def pop(self) -> td.TensorDict:
        return self._list.pop()

    @typing.override
    def __getitems__(self, index: list[int], /):
        # Check index out of bounds.
        idx: IntArray = np.asarray(index)
        if any(idx < -len(self)) or any(idx >= len(self)):
            violation = np.concat([idx[idx < -len(self)], idx[idx >= len(self)]])
            raise IndexError(
                f"Part of the index: {violation=} out of bounds for {len(self)=}."
            )

        # Convert to positive.
        return self.__getitems(idx % len(self))

    def __getitems(self, idx: IntArray, /) -> td.TensorDict:
        assert all(idx >= 0)
        assert all(idx < len(self))

        # Which tensordict to use in `self.tensordicts`.
        td_idx = np.searchsorted(self._cumsum_len, idx, side="right")
        assert td_idx.shape == idx.shape

        # How many elements are in the partitions prior to the current.
        prior_elements = np.roll(self._cumsum_len, 1)
        prior_elements[0] = 0

        # Index in partition = original index - elements in prior partitions.
        idx_in_part = idx - prior_elements[td_idx]

        # `td.TensorDict` that each index would correspond to.
        td_for_idx: list[td.TensorDict] = [self._list[t] for t in td_idx]

        assert len(idx_in_part) == len(td_for_idx)

        # Get each from partition, sequentially (maybe improve this in the future).
        chunks: list[td.TensorDict] = []
        for tdict, part_idx in zip(td_for_idx, idx_in_part.tolist()):
            assert -len(tdict) <= part_idx < len(tdict), {
                "index for sub partition": part_idx,
                "tensordict's length": len(tdict),
            }
            chunks.append(tdict[part_idx : part_idx + 1])

        return td.cat(chunks).auto_batch_size_()

    @property
    def _cumsum_len(self):
        return np.cumsum([len(d) for d in self._list])

    @property
    @typing.override
    def attrs(self) -> AttrDict:
        return self._attrs

    @functools.cached_property
    def _attrs(self):
        attrs = [AttrDict.parse(chunk) for chunk in self._list]

        if len({*attrs}) == 1:
            return attrs[0]

        raise ValueError("`td.TensorDict` should convert to the same attrs.")
