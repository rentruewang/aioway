# Copyright (c) AIoWay Authors - All Rights Reserved

"Metadata for torch operators / functions."

import contextlib as ctxl
import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

__all__ = ["Fn", "Thunk", "FnStack", "pretty_function_call"]

_PENDING = object()
"The object signifying a status of pending. This is a `object()` s.t. `FnCache` can store `None`."


class Fn(typing.Protocol):
    def do(self):
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
            self.__result = self.func(*self.args, **self.kwargs)

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
        pretty = pretty_function_call(
            self.func.__qualname__,
            *self.args,
            **self.kwargs,
        )

        if self.done:
            pretty = f"{pretty} -> {self.__result!r}"

        return pretty

    @property
    def done(self) -> bool:
        "Returns if the cache is previously called."

        return self.__result is not _PENDING


@dcls.dataclass
class FnStack[F: Fn]:
    """
    `FnStack` is the tracker for `Fn`, storing `Fn`s that are currengly `do()`-ing.
    """

    stack: list[F]
    """
    The stack that is currently in scope.
    """

    def __bool__(self):
        return bool(len(self))

    def __len__(self):
        return len(self.stack)

    @typing.overload
    def __getitem__(self, idx: int) -> F: ...

    @typing.overload
    def __getitem__(self, idx: slice) -> typing.Self: ...

    @typing.no_type_check
    def __getitem__(self, idx):
        match idx:
            case int():
                return self.stack[idx]
            case slice():
                return type(self)(self.stack[idx])

        raise TypeError(type(idx))

    def append(self, fn: F) -> None:
        self.stack.append(fn)

    def pop(self) -> F:
        return self.stack.pop()

    @ctxl.contextmanager
    def track_scope(self, fn: F):
        try:
            yield
        finally:
            self.pop()

    # def track_func(self, function: )


def pretty_function_call(func: str, *args: typing.Any, **kwargs: typing.Any) -> str:
    args_builder: list[str] = []

    # Add positional arguments.
    if args:
        args_builder.extend(f"{arg!r}" for arg in args)

    # Add keyword arguments.
    if kwargs:
        args_builder.extend(f"{k!s}={v!r}" for k, v in kwargs.items())

    args_str = ", ".join(args_builder)
    return f"{func}({args_str})"
