# Copyright (c) AIoWay Authors - All Rights Reserved

"The base classes for modes."

import abc
import contextlib as ctxl
import dataclasses as dcls
import logging
import typing

from aioway._utils import Stack

__all__ = ["OnOffCtx", "OnOffStack"]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass
class OnOffCtx(abc.ABC):
    """
    `OnOffCtx` is a mixin class that gives the subclasses a toggle.
    """

    STACK: typing.ClassVar[OnOffStack[typing.Self]]
    "The stack."

    _: dcls.KW_ONLY

    on: bool = True
    "The toggle to control whether or not to run the current mode."

    @abc.abstractmethod
    def enter(self) -> typing.ContextManager[typing.Self]:
        """
        The context manager that can be entered, and will be constrained by `self.on`.

        I'm using this function as public API because I don't like `__enter__`, `__exit__`,
        which is much less elegant than `ctxl.contextmanager` (I know it's necessary).
        """

        raise NotImplementedError

    @ctxl.contextmanager
    def switch(self, on: bool, /):
        "Switch to `on` in the scope (can be overwritten)."

        before = self.on
        self.on = on
        try:
            yield
        finally:
            self.on = before


class OnOffStack[T: OnOffCtx](Stack[T]):
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
