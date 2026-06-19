# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Session` interface."

import abc
import contextlib as ctxl
import dataclasses as dcls
import inspect
import typing
from collections import abc as cabc

__all__ = ["Session"]


@dcls.dataclass(frozen=False, slots=True)
class _RootSession[S: Session]:
    """
    The root session simplifies the type checking code in `Session`.
    It represents the default scope.
    """

    _child: S | None = dcls.field(default=None, init=False)
    """
    The child session.
    """

    def __repr__(self):
        return repr(self._child)


class Session[T](abc.ABC):
    """
    The cost session. Use the `track_cost()` function to track the costs in a new scope.
    Providing `.sum()` function for summarization of costs.
    Not thread safe, but efficient in single threading context.
    """

    _active: typing.ClassVar[typing.Any] = _RootSession()
    "The currently active session."

    def __init__(self) -> None:
        self._parent: _RootSession[typing.Self] | typing.Self = _RootSession()
        "The parent session."

        self._child: typing.Self | None = None
        "The child session."

        self.__is_active: bool = False
        "Whether or not the session is turned on."

    @typing.final
    @ctxl.contextmanager
    def __call__(self) -> cabc.Generator[T]:
        with self.__set_active_flag(), self._set_parent_active(), self.do() as t:
            yield t

    @abc.abstractmethod
    def do(self) -> typing.ContextManager[T]:
        raise NotImplementedError

    @ctxl.contextmanager
    def _set_parent_active(self) -> cabc.Generator[None]:
        cls = type(self)

        if cls._active._child is not None:
            raise AssertionError(
                f"Parent already has a child. {cls._active} failed to clean up."
            )

        # Set `self` to be a child of `cls._active`, and use `self` as new `cls._active`.
        old_parent, self._parent, cls._active = self._parent, cls._active, self

        try:
            yield

        # After done, hand the `_active` position back to parent.
        finally:
            self._parent, cls._active = old_parent, self._parent

    @ctxl.contextmanager
    def __set_active_flag(self):
        if self.__is_active:
            raise RuntimeError(f"Entering the same session {self} twice.")

        self.__is_active = True

        try:
            yield
        finally:
            self.__is_active = False

    @property
    def is_active(self) -> bool:
        """
        Whether we are in the scope of this current `Session`.
        """

        if type(self)._active is self:
            return True

        if self._child is not None:
            return self._child.is_active

        return False

    def ancestors(self) -> cabc.Iterator[typing.Self]:
        """
        Yield the `Session`'s ancestors. Includes `self`.
        """

        if isinstance(self._parent, _RootSession):
            return

        # Yield parent's ancestors first s.t. it will be in order of root to leaf.
        yield from self._parent.ancestors()
        yield self

    def decendants(self) -> cabc.Iterator[typing.Self]:
        """
        Yield the `Session`'s decendants. Includes `self`.
        """

        if self._child is None:
            return

        yield self
        yield from self._child.decendants()

    @classmethod
    def current(cls) -> typing.Self | None:
        """
        Returns the most recently active `Session`.
        """

        if inspect.isabstract(cls):
            raise RuntimeError(f"{cls=} is abstract! `.current()` not supported.")

        if not isinstance(cls._active, cls):
            assert isinstance(cls._active, _RootSession), cls._active
            return None

        return cls._active

    @classmethod
    def active_sessions(cls) -> cabc.Iterator[typing.Self]:
        """
        Get all the `Session` of `cls` that is in scope.
        """

        active: typing.Self | _RootSession = cls._active

        if isinstance(active, _RootSession):
            return

        assert isinstance(active, Session)

        for ancestor in active.ancestors():
            assert isinstance(ancestor, cls)
            yield ancestor
