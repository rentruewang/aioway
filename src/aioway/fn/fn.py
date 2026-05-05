# Copyright (c) AIoWay Authors - All Rights Reserved

"Metadata for torch operators / functions."

import abc
import contextlib as ctxl
import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

from aioway._common import render_fcall, render_fcall_done

__all__ = ["Fn", "Thunk", "FnStack"]

_PENDING = object()
"The object signifying a status of pending. This is a `object()` s.t. `FnCache` can store `None`."


class Fn(abc.ABC):
    """
    `Fn` is the base class for delayed computation.

    `Fn.do` executes the computation, `Fn` base class itself does not make any more assumption.
    """

    @abc.abstractmethod
    def do(self) -> typing.Any:
        """
        Execute the computation.
        """

        raise NotImplementedError


class Thunk(Fn):
    """
    The thunk for any function, handles both pretty printing and storing the result.

    `Fn` has advantage over `functools.cache`, `functools.cached_property`,
    and having a saved `.__result` member for instance, by being the least assuming.

    `functools.cache` assumes that `self` is hashable.
    `functools.cached_property` cannot inspect whether we have evaluated it or not.
    `.__result` member assumes subclass calls `__init__` properly.

    Storing a `Fn` in a `functools.cached_property` means `self` can be unhashable,
    the item can be inspected, and subclasses do not need to call `__init__`.

    Like Haskell's thunks, once evaluated,
    the value is stored in the `Fn` itself and never re-evaluated.
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

        if self.done:
            return render_fcall_done(self.func, self.__result, *args, **kwargs)
        else:
            return render_fcall(self.func, *args, **kwargs)

    def result(self) -> object:
        "Get the result. If not done, an error is raised."

        if self.__result is None:
            raise RuntimeError("Still waiting for the result!")

        return self.__result

    @property
    def done(self) -> bool:
        "Returns if the cache is previously called."

        return self.__result is not _PENDING


@dcls.dataclass
class FnStack[F: Fn]:
    """
    `FnStack` is the tracker for `Fn`, storing `Fn`s that are currengly `do()`-ing.
    """

    stack: list[F] = dcls.field(default_factory=list)
    """
    The stack that is currently in scope.
    """

    def __bool__(self) -> bool:
        return bool(len(self))

    def __len__(self) -> int:
        return len(self.stack)

    @typing.overload
    def __getitem__(self, idx: int) -> F: ...

    @typing.overload
    def __getitem__(self, idx: slice[int]) -> typing.Self: ...

    def __getitem__(self, idx: int | slice[int]):
        match idx:
            case int():
                return self.stack[idx]
            case slice():
                return type(self)(self.stack[idx])

        raise TypeError(type(idx))

    def __iter__(self) -> cabc.Generator[F]:
        yield from self.stack

    def top(self) -> F:
        return self.stack[-1]

    def append(self, fn: F) -> None:
        self.stack.append(fn)

    def pop(self) -> F:
        return self.stack.pop()

    @ctxl.contextmanager
    def track(self, fn: F):
        self.append(fn)
        try:
            yield
        finally:
            _ = self.pop()
