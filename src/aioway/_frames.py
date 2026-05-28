# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Frame` interface."

import abc
import dataclasses as dcls
import typing

import tensordict as td
import torch
from torch.utils import data

from aioway.schemas import Attr, AttrDict

__all__ = ["Frame", "TensorFrame", "TdictFrame", "frame_dcls"]


@typing.dataclass_transform(frozen_default=True)
def frame_dcls(cls):
    return dcls.dataclass(frozen=True)(cls)


class _HasGetItem(typing.Protocol):
    def __getitem__(self, idx: int) -> typing.Self: ...


@frame_dcls
class Frame[T: _HasGetItem](data.Dataset[T], abc.ABC):
    """
    `Frame` represents a set of heterogenious data stored in memory,
    it is one of the main physical abstractions in `aioway` to represent eager computation.

    Each item retrieved from `Frame` is a minibatch of data.
    """

    def __bool__(self) -> bool:
        return bool(len(self))

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Get the number of items (rows) in the current dataframe.
        """

        raise NotImplementedError

    @typing.final
    def __getitem__(self, idx: int) -> T:
        if not isinstance(idx, int):
            raise TypeError(f"__getitem__ only accepts `int`, got {idx=}")
        result = self.__getitems__([idx])
        return result[0]

    @abc.abstractmethod
    def __getitems__(self, idx: list[int], /) -> T:
        raise NotImplementedError


@frame_dcls
class TensorFrame(Frame[torch.Tensor], abc.ABC):
    "A `Frame[torch.Tensor]`."

    @property
    @abc.abstractmethod
    def attr(self) -> Attr:
        "The schema of the current frame."

        raise NotImplementedError


@frame_dcls
class TdictFrame(Frame[td.TensorDict], abc.ABC):
    "A `Frame[td.TensorDict]`."

    @property
    @abc.abstractmethod
    def attrs(self) -> AttrDict:
        "The schema of the current frame."

        raise NotImplementedError

    def column(self, col: str):
        from aioway.relalg import FrameColumnView

        return FrameColumnView(self, col)

    def select(self, *cols: str):
        from aioway.relalg import FrameSelectView

        return FrameSelectView(self, cols)
