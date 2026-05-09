# Copyright (c) AIoWay Authors - All Rights Reserved

"Metadata for torch operators / functions."

import abc
import functools
import typing
from collections import abc as cabc

from aioway._common import render_fcall

__all__ = ["Fn", "Thunk"]

_PENDING = object()
"The object signifying a status of pending. This is a `object()` s.t. `FnCache` can store `None`."


@typing.runtime_checkable
class Fn(typing.Protocol):
    """
    `Fn` is the base class for delayed computation, in a single batch (a single pass).

    It stands for [f]unction [n]ode, or short for function.

    `Fn.do` executes the computation, `Fn` base class itself does not make any more assumption.
    """

    @abc.abstractmethod
    def do(self) -> typing.Any:
        """
        Execute the computation.
        """


class Thunk:
    """
    The thunk for any function, handles both pretty printing and storing the result.

    `Thunk` has advantage over `functools.cache`, `functools.cached_property`,
    and having a saved `.__result` member for instance, by being the least assuming.

    `functools.cache` assumes that `self` is hashable.
    `functools.cached_property` cannot inspect whether we have evaluated it or not.
    `.__result` member assumes subclass calls `__init__` properly.

    Storing a `Thunk` in a `functools.cached_property` means `self` can be unhashable,
    the item can be inspected, and subclasses do not need to call `__init__`.

    Like Haskell's thunks, once evaluated,
    the value is stored in the `Thunk` itself and never re-evaluated.
    The value shall be gone during GC.

    I was going to go for `Op` but it's used a lot in `torch`.
    """

    def __init__(
        self,
        func: cabc.Callable[..., typing.Any],
        /,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        if not callable(func):
            raise ValueError("Cannot create thunk for function that isn't callable.")

        self._func = func
        self._args = args
        self._kwargs = kwargs

        self.__result: object = _PENDING
        "If not evaluated, it's pending."

    @typing.override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Thunk):
            return (
                True
                and self.func == other.func
                and self.args == other.args
                and self.kwargs == other.kwargs
            )

        return NotImplemented

    def do(self) -> typing.Any:
        """
        Call and cache the function.
        """

        if self.__result is _PENDING:
            try:
                self.__result = self.func(*self.args, **self.kwargs)
            except Exception as e:
                raise RuntimeError(f"Thunk: {self!r} evaluation failed.") from e

        return self.__result

    @typing.override
    def __repr__(self) -> str:
        return self.__string

    @typing.override
    def __str__(self) -> str:
        return self.__string

    @property
    def thunk(self) -> Thunk:
        return self

    @property
    def func(self):
        return self._func

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs

    @functools.cached_property
    def __string(self) -> str:
        args, kwargs = self.args, self.kwargs

        fcall = render_fcall(self.func, *args, **kwargs)

        if self.done:
            fcall += f" -> {self.__result}"

        return fcall

    def result(self) -> object:
        "Get the result. If not done, an error is raised."

        if self.__result is None:
            raise RuntimeError("Still waiting for the result!")

        return self.__result

    @property
    def done(self) -> bool:
        "Returns if the cache is previously called."

        return self.__result is not _PENDING
