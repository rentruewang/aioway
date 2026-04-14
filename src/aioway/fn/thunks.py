# Copyright (c) AIoWay Authors - All Rights Reserved

import functools
import typing
from collections import abc as cabc

__all__ = ["Thunk", "TorchDispatchThunk"]

_PENDING = object()
"The object signifying a status of pending. This is a `object()` s.t. `FnCache` can store `None`."


class Thunk[**P, T]:
    """
    The thunk for any function.

    The reason we use this boilerplate over directly using `functools.cache`,
    `functools.cached_property`, or having a saved `.__result` member for instance,
    is because this is the least assuming.

    `functools.cache` assumes that `self` is hashable.
    `functools.cached_property` cannot inspect whether we have evaluated it or not.
    `.__result` member assumes subclass calls `__init__` properly.

    Since this is saved in a `functools.cached_property`, it can be used on unhashable types,
    yet support inspecting whether we called it or not, and does not need to call `__init__`.
    """

    def __init__(
        self, func: cabc.Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
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

    def __call__(self) -> T:
        """
        Call and cache the function.
        """

        if self.__result is _PENDING:
            self.__result = self.func(*self.args, **self.kwargs)

        return typing.cast(T, self.__result)

    @typing.override
    def __repr__(self) -> str:
        return self.__string

    @typing.override
    def __str__(self) -> str:
        return self.__string

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
        return _format_function_as_str(self.func, *self.args, **self.kwargs)

    @property
    def done(self) -> bool:
        "Returns if the cache is previously called."

        return self.__result is not _PENDING


class TorchDispatchThunk[**P, T](Thunk[P, T]):
    def __init__(
        self,
        func: cabc.Callable[P, T],
        types: tuple[type, ...],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        super().__init__(func, *args, **kwargs)
        self._types = types

    @property
    def types(self):
        return self._types


def _format_function_as_str(
    func: cabc.Callable[..., typing.Any], *args: typing.Any, **kwargs: typing.Any
) -> str:
    args_builder: list[str] = []

    # Add positional arguments.
    if args:
        args_builder.extend(f"{arg!r}" for arg in args)

    # Add keyword arguments.
    if kwargs:
        args_builder.extend(f"{k!s}={v!r}" for k, v in kwargs.items())

    args_str = ", ".join(args_builder)
    func_str = func.__qualname__
    return f"{func_str}({args_str})"
