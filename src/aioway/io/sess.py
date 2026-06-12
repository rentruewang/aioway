# Copyright (c) AIoWay Authors - All Rights Reserved

"The session for the datasets."

import contextlib as ctxl
import typing

from aioway._utils import Stack
from aioway.sess import Session

from .dsets import Frame, Sink, Stream

__all__ = ["StreamSession", "SinkSession", "FrameSession"]

_STREAMS: Stack[Stream] = Stack()
"The streams that are in scope."

_FRAMES: Stack[Frame] = Stack()
"The frames that are in scope."

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


class StreamSession(_DsetSession[Stream]):
    "The dataset session for streams."

    STACK: typing.ClassVar = _STREAMS


class SinkSession(_DsetSession[Sink]):
    "The dataset session for sinks."

    STACK: typing.ClassVar = _SINKS


class FrameSession(_DsetSession[Frame]):
    "The dataset session for frames."

    STACK: typing.ClassVar = _FRAMES
