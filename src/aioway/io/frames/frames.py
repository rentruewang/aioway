# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Frame` interface."

import abc
import dataclasses as dcls
import typing

import numpy as np
import tensordict as td
from torch.utils import data

from aioway._utils import is_list_of
from aioway.dsets import Dataset, DatasetColumnView, DatasetSelectView, DatasetViewTypes
from aioway.schemas import AttrDict

__all__ = ["FrameDict"]


@dcls.dataclass(frozen=True)
class FrameDict(Dataset, data.Dataset[td.TensorDict], abc.ABC):
    """
    `FrameDict` represents a set of heterogenious data stored in memory,
    it is one of the main physical abstractions in `aioway` to represent eager computation.

    Think of it as a normal `Sequence` of `td.TensorDict`,
    where computation happens eagerly, imperatively, and the result is stored in memory.

    Each `td.TensorDict` retrieved from `FrameDict` is a minibatch of data.

    Similar to `Dataset`, but only allows retrieving a batch at a time.
    To get a single item, retrieve a batch of size 1.

    For simplicity of API, this class does not support `__getitem__(int)`,
    as that is not needed because all index access should be batched (slice, arrays),
    and unecessarily makes implementation duplicate.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Get the number of items (rows) in the current dataframe.
        """

    def __bool__(self) -> bool:
        return bool(len(self))

    @typing.overload
    def __getitem__(self, idx: str) -> DatasetColumnView[typing.Self]: ...

    @typing.overload
    def __getitem__(self, idx: list[str]) -> DatasetSelectView[typing.Self]: ...

    @typing.overload
    def __getitem__(
        self, idx: int | slice | list[int] | np.ndarray
    ) -> td.TensorDict: ...

    @typing.final
    @typing.override
    def __getitem__(self, idx, /):
        """
        Get individual items from the current `Frame`.

        Args:
            idx:
                Index to the current `Frame`.
                If it is a `str` or `list[str]`, it is considered a `Table` operation.

                For indexing operations, index type must be a `slice`,
                or `list[int]`, or a numpy array.
                Should be in the range `[-len, len)`.

        Returns:
            A `TensorDict` representing a batch of data.
        """

        if isinstance(idx, str):
            return self.column(idx)

        if is_list_of(str)(idx):
            return self.select(*idx)

        # If slice, convert to `range(len(self))[idx]`.
        # This will be the same length as the output list,
        # so it's ok that `NDArray` is less efficient than `slice`.
        it: range | list[int] | np.ndarray
        if isinstance(idx, slice):
            it = range(len(self))[idx]
        else:
            it = idx

        arr: np.ndarray = np.asarray(it)
        arr = self._check_idx(arr)

        item = self._getitems_batch(arr.tolist())

        return item

    __geitems__ = __getitem__

    @property
    @abc.abstractmethod
    def attrs(self) -> AttrDict:
        "The schema of the current frame."

        raise NotImplementedError

    @abc.abstractmethod
    def _getitems_batch(self, idx: list[int]) -> td.TensorDict:
        raise NotImplementedError

    @classmethod
    @typing.override
    def view_types(cls):
        from .views import FrameColumnView, FrameSelectView

        return DatasetViewTypes(column=FrameColumnView, select=FrameSelectView)

    def _check_idx(self, idx: np.ndarray, /) -> np.ndarray:
        "Check if the index is valid, and then remap the index to be positive."

        length = len(self)

        if np.all(idx < -length) or np.all(idx >= length):
            raise IndexError(
                f"Index must be in the range `[-{length}, {length})`, but got {idx=}"
            )

        return idx % length
