# Copyright (c) AIoWay Authors - All Rights Reserved

"The interface for the `io` package."

import abc
import typing
from collections import abc as cabc

import tensordict as td
import torch
from torch.utils import data as dutils
from torchrl import data as rldata

__all__ = [
    "Dset",
    "Stream",
    "Frame",
    "TensorStream",
    "TdictStream",
    "TensorFrame",
    "TdictFrame",
]


class Dset[T](dutils.Dataset[T]):
    """
    The base class for I/O.
    """

    @typing.final
    def __post_init__(self) -> None:
        self._setup()
        self._register()

    def __space__(self) -> rldata.TensorSpec:
        return NotImplemented

    def _setup(self) -> None:
        """
        Subclass should overwrite this function to perform setup. Can raise errors.
        """

    def _register(self) -> None:
        """
        Register the instance to a session.
        """

        from .sess import DsetSession

        if sess := DsetSession.current():
            sess.push(self)


class Stream[T = typing.Any](Dset[T], dutils.IterableDataset[T], abc.ABC):
    """
    `Stream` represents a set of sequential data stored somewhere.
    Each item is a single row of data.
    """

    @abc.abstractmethod
    def __iter__(self) -> cabc.Iterator[T]:
        raise NotImplementedError


class TensorStream(Stream[torch.Tensor], abc.ABC):
    """
    A `TensorStream` is a `Stream` of `torch.Tensor`s.
    """


class TdictStream(Stream[td.TensorDict], abc.ABC):
    """
    A `TdictStream` is a `Stream` of `torch.Tensor`s.
    """


class Frame[T = typing.Any](Dset[T], dutils.Dataset[T], abc.ABC):
    """
    `Frame` is a `Stream` that supports random access.
    Each item retrieved from `Frame` is a single row of data.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Get the number of items (rows) in the current dataframe.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, index: int) -> T:
        """
        Get 1 item.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def __getitems__(self, index: list[int], /) -> T:
        """
        Get multiple items.
        """

        raise NotImplementedError


class TensorFrame(Frame[torch.Tensor], abc.ABC):
    """
    A `torch.Tensor` dataset that supports random access.
    """


class TdictFrame(Frame[td.TensorDict], abc.ABC):
    """
    A dataset of `td.TensorDict` that supports random access.
    """
