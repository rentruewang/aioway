# Copyright (c) AIoWay Authors - All Rights Reserved

"The session for the datasets."

import contextlib as ctxl
import typing

from aioway._utils import Stack
from aioway.sess import Session

from .dsets import Dset, Sink

__all__ = ["DsetSession", "SinkSession"]

_DATASETS: Stack[Dset] = Stack()
"The datasets that are in scope."


_SINKS: Stack[Sink] = Stack()
"The list of sinks."


class _DsetSession[T](Session):
    STACK: typing.ClassVar[Stack[T]]
    """
    The stack for the current session (must be overwritten in subclass).
    """

    def __init__(self):
        super().__init__()

        self._before_len = len(self.STACK)

    def __len__(self) -> int:
        return len(self.STACK) - self._before_len

    def __getitem__(self, idx: int, /):
        return self.STACK[idx - self._before_len]

    def push(self, item: T, /) -> None:
        self.STACK.append(item)

    @typing.override
    @ctxl.contextmanager
    def do(self):
        try:
            yield self
        finally:
            self.STACK.truncate(self._before_len)


class DsetSession(_DsetSession[Dset]):
    "The dataset session for streams and frames."

    STACK: typing.ClassVar = _DATASETS


class SinkSession(_DsetSession[Sink]):
    "The dataset session for sinks."

    STACK: typing.ClassVar = _SINKS
