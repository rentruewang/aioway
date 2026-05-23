# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Stream` interfaces live here."

import abc
import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

import tensordict as td

from aioway.specs import AttrSet

from ..datasets import Dataset, DatasetViewTypes

__all__ = ["Stream", "StreamState", "Stream", "Stream", "Stream"]


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


@dcls.dataclass(frozen=True)
class Stream(cabc.Iterator[td.TensorDict], Dataset, abc.ABC):
    """
    `Stream` produces a stream of batches of data, in the form of `TensorDict`s,
    everytime `__next__` is called on it, a `TensorDict` is yielded.

    `Stream` is a stateful operation, compared to the previous implementations,
    it is an external iterator, supporting state inspection, simplifying debugging.
    """

    __match_args__: typing.ClassVar[tuple[str, ...]]
    """
    A `Stream` should be able to be decomposed with `match` statements.
    """

    @typing.override
    def __iter__(self) -> typing.Self:
        """
        `__iter__` allows `Stream`s to be used in `for` loops.

        As it returns `self`, re-use carefully.
        """

        return self

    @typing.final
    @typing.override
    def __next__(self) -> td.TensorDict:
        """
        `__next__` allows `Stream`s to be used in `for` loops.
        """

        result = self._next()
        self.state.step()
        return result

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """
        The length of the current `Stream`.
        Does not change when the `Stream` is being iterated over.
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def attrs(self) -> AttrSet:
        """
        The schema for the current `Stream`.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def _next(self) -> td.TensorDict:
        """
        Compute the next batch (a `td.TensorDict`).

        An exception raised here would be translated to `StopIteration`.

        Note:

            The name `read` is inspired by the original release of clickhouse.
            I think this is a great name for a as a method on `Stream` to poll data,
            giving it a natural feeling like how we normally work with files.

        """

        raise NotImplementedError

    @abc.abstractmethod
    def _inputs(self) -> tuple[Stream, ...]:
        """
        `Stream`'s children, the dependent `Stream`s that would also be evaluated
        when calling `__next__` on the current `Stream`.
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

    @classmethod
    @typing.override
    def view_types(cls):
        from .views import StreamColumnView, StreamSelectView

        return DatasetViewTypes(column=StreamColumnView, select=StreamSelectView)
