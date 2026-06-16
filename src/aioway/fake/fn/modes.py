# Copyright (c) AIoWay Authors - All Rights Reserved

"The base classes for modes."

import abc
import contextlib as ctxl
import dataclasses as dcls
import logging
import typing
from collections import abc as cabc

from aioway._fn import TorchThunk
from aioway._utils import Stack

__all__ = ["OnOffCtx", "OnOffStack"]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass
class Mode[T: TorchThunk = TorchThunk, V = object](abc.ABC):
    """
    `Mode` is a mixin class that gives the subclasses a toggle.
    """

    STACK: typing.ClassVar[ModeStack]
    "The stack."

    _: dcls.KW_ONLY

    on: bool = True
    "The toggle to control whether or not to run the current mode."

    @ctxl.contextmanager
    def __call__(self) -> cabc.Generator[typing.Self]:
        """
        The context manager that can be entered, and will be constrained by `self.on`.

        I'm using this function as public API because I don't like `__enter__`, `__exit__`,
        which is much less elegant than `ctxl.contextmanager` (I know it's necessary).
        """

        with self.STACK.hold(self), self.enter():
            yield self

    @abc.abstractmethod
    def run(self, thunk: T, /) -> V:
        """
        The overriding function that customizes `thunk()`.
        Calling `thunk()` should run the next `Mode.run` until the `STACK` is exhausted,

        """

        raise NotImplementedError

    @ctxl.contextmanager
    def enter(self) -> cabc.Generator[None]:
        yield

    @ctxl.contextmanager
    def switch(self, on: bool, /):
        "Switch to `on` in the scope (can be overwritten)."

        before = self.on
        self.on = on
        try:
            yield
        finally:
            self.on = before


class ModeStack[T: Mode[typing.Any, typing.Any]](Stack[T]):
    """
    `OnOffStack` provides additional utilites to decide when to turn on or off.
    """

    @ctxl.contextmanager
    def switch(self, on: bool | list[bool], /):
        """
        Temporarily set the `on` switch to the value given.
        """

        before = self.on
        self.on = on

        try:
            yield
        finally:
            self.on = before

    @property
    def on(self) -> list[bool]:
        "Get the on off values."

        return [frame.on for frame in self]

    @on.setter
    def on(self, to: bool | list[bool]) -> None:
        LOGGER.debug("Current stack %s", self)
        LOGGER.debug("Setting to %s", to)

        if isinstance(to, bool):
            to = [to] * len(self)

        if len(to) != len(self):
            raise ValueError(f"Value {to=} should have equal length with {self=}.")

        for frame, val in zip(self, to):
            frame.on = val

        LOGGER.debug("Status after setting %s", self)
