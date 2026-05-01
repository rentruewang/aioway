# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

__all__ = ["render_fcall", "render_fcall_done"]

type FunctionLike = str | cabc.Callable[..., typing.Any]


def render_fcall(func: FunctionLike, *args: typing.Any, **kwargs: typing.Any) -> str:
    args_builder: list[str] = []

    # Add positional arguments.
    if args:
        args_builder.extend(f"{arg!r}" for arg in args)

    # Add keyword arguments.
    if kwargs:
        args_builder.extend(f"{k!s}={v!r}" for k, v in kwargs.items())

    args_str = ", ".join(args_builder)
    return f"{func!r}({args_str})"


def render_fcall_done(
    func: FunctionLike, returns: typing.Any, *args: typing.Any, **kwargs: typing.Any
) -> str:
    lhs = render_fcall(func, *args, **kwargs)
    return f"{lhs!s} -> {returns!r}"
