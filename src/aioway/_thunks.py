# Copyright (c) AIoWay Authors - All Rights Reserved

"Metadata for torch operators / functions."

import functools
import logging
import typing
from collections import abc as cabc

from aioway._utils import render_fcall

__all__ = ["Thunk", "AnyThunk"]

LOGGER = logging.getLogger(__name__)


@typing.runtime_checkable
class Thunk[T = object](typing.Protocol):
    """
    `Thunk` is the base class for delayed computation, in a single batch (a single pass).

    It stands for [f]unction [n]ode, or short for function.

    `Thunk()` executes the computation, `Thunk` base class itself does not make any more assumption.
    """

    def __call__(self) -> T:
        """
        Execute the computation.
        """

        raise NotImplementedError


class AnyThunk:
    """
    The thunk for any function, handles both pretty printing and storing the result.

    `AnyThunk` has advantage over `functools.cache`, `functools.cached_property`,
    and having a saved `.__result` member for instance, by being the least assuming.

    `functools.cache` assumes that `self` is hashable.
    `functools.cached_property` cannot inspect whether we have evaluated it or not.
    `.__result` member assumes subclass calls `__init__` properly.

    Storing a `AnyThunk` in a `functools.cached_property` means `self` can be unhashable,
    the item can be inspected, and subclasses do not need to call `__init__`.

    Like Haskell's thunks, once evaluated,
    the value is stored in the `AnyThunk` itself and never re-evaluated.
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

    @typing.override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, AnyThunk):
            return (
                True
                and self.func == other.func
                and self.args == other.args
                and self.kwargs == other.kwargs
            )

        return NotImplemented

    def __call__(self) -> object:
        return self.func(*self.args, **self.kwargs)

    @typing.override
    def __repr__(self) -> str:
        return self.__string

    @typing.override
    def __str__(self) -> str:
        return self.__string

    @property
    def thunk(self) -> AnyThunk:
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

        return render_fcall(self.func, *args, **kwargs)
