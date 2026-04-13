# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

from .typing import is_dict_of_str_to

__all__ = ["format_function", "dcls_no_eq", "dcls_no_eq_no_repr"]


@dcls.dataclass(frozen=True)
class _FuncFormat:
    """
    A custom object that lazily formats the function, to avoid unecessary computation.
    """

    func: cabc.Callable[..., typing.Any]
    "The function. Must be callable."

    args: tuple[typing.Any, ...]
    kwargs: dict[str, typing.Any]

    def __post_init__(self) -> None:
        if not callable(self.func):
            raise ValueError("Cannot format function that isn't callable.")

        if not isinstance(self.args, tuple):
            raise ValueError(f"Args: {self.args!r} is not valid.")

        if not is_dict_of_str_to(object)(self.kwargs):
            raise ValueError(f"Kwargs: {self.kwargs!r} is not valid.")

    @typing.override
    def __str__(self) -> str:
        return self.__string

    @functools.cached_property
    def __string(self) -> str:
        return _format_function_as_str(self.func, *self.args, **self.kwargs)


def format_function(
    func: cabc.Callable[..., typing.Any], *args: typing.Any, **kwargs: typing.Any
):
    """
    Formats the function into readable string, mimicking signature in python.

    Returns a custom object that lazily formats the function call to string.
    """

    return _FuncFormat(func, args, kwargs)


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


@typing.dataclass_transform(eq_default=False)
def dcls_no_eq[T: type](cls: T) -> T:
    result: typing.Any = dcls.dataclass(eq=False)(cls)
    return result


@typing.dataclass_transform(eq_default=False)
def dcls_no_eq_no_repr[T: type](cls: T) -> T:
    result: typing.Any = dcls.dataclass(eq=False, repr=False)(cls)
    return result
