# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Stream` interfaces live here."

import abc
import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway._utils import decomp_dcls_members
from aioway.attrs import Attr, AttrDict

__all__ = ["StreamState", "Stream", "TensorStream", "TdictStream", "stream_dcls"]


@typing.dataclass_transform(frozen_default=True)
def stream_dcls(cls):
    return dcls.dataclass(frozen=True)(cls)


@dcls.dataclass
class StreamState:
    """
    The mutable stream state.

    This is created because `Stream` subclasses from a frozen `dataclass`,
    so the stream state is created to manage mutable parts of the `Stream`.

    Subclasses of `Stream` should also subclass from `StreamState`.
    """

    idx: int = 0
    "How many steps have been called."

    def step(self):
        self.idx += 1

    @property
    def started(self) -> bool:
        """
        Shortcut function to check if `self.idx == 0`.
        """

        return self.idx != 0


@stream_dcls
class Stream[T](cabc.Iterator[T], abc.ABC):
    """
    The base class for `Stream` and `StreamDict`,
    yielding batches of `torch.Tensor` and `td.TensorDict`, respectively.

    It's a stateful operation, using `StreamState`.
    """

    __match_args__: typing.ClassVar[tuple[str, ...]]
    """
    A `Stream` should be able to be decomposed with `match` statements.
    """

    @typing.override
    def __iter__(self) -> typing.Self:
        return self

    @typing.final
    @typing.override
    def __next__(self) -> T:
        result = self.read()
        self.state.step()
        return result

    @abc.abstractmethod
    def read(self) -> T:
        """
        Compute the next batch.

        An exception raised here would be translated to `StopIteration`.
        """

        raise NotImplementedError

    @functools.cached_property
    def state(self) -> StreamState:
        """
        The state of the stream. Should be a field, but a `cached_property`,
        because if it has a default value it would make subclassing difficult.
        """
        return StreamState()

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """
        The length of the current `Stream`.
        Does not change when the `Stream` is being iterated over.
        """

        raise NotImplementedError

    @property
    def idx(self) -> int:
        """
        The number of batches completed..
        """

        return self.state.idx

    @property
    def started(self) -> bool:
        """
        Shortcut function to check if `self.idx == 0`.
        """

        return self.state.started

    @typing.final
    def inputs(self) -> cabc.Iterator[Stream[T]]:
        """
        The input of the current `Stream`.
        """

        yield from decomp_dcls_members(self, Stream)


@stream_dcls
class TensorStream(Stream[torch.Tensor], abc.ABC):
    """
    An iterator of batches of `torch.Tensor`.
    """

    @property
    @abc.abstractmethod
    def attr(self) -> Attr:
        """
        The schema for the current `Stream`.
        """

        raise NotImplementedError


@stream_dcls
class TdictStream(Stream[td.TensorDict], abc.ABC):
    """
    An iterator of batches of `td.TensorDict`.
    """

    @property
    @abc.abstractmethod
    def attrs(self) -> AttrDict:
        """
        The schema for the current `Stream`.
        """

        raise NotImplementedError

    def column(self, col: str) -> TensorStream:
        from aioway.relalg import StreamColumnView

        return StreamColumnView(self, col)

    def select(self, *cols: str) -> TdictStream:
        from aioway.relalg import StreamSelectView

        return StreamSelectView(self, cols)
