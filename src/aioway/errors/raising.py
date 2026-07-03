# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
from collections import abc as cabc

__all__ = ["re_raise_func", "re_raise_scope"]


@ctxl.contextmanager
def re_raise_scope(
    before: type[BaseException], after: type[BaseException], /
) -> cabc.Generator[None]:
    try:
        yield
    except before as exc:
        raise after from exc


def re_raise_func(before: type[BaseException], after: type[BaseException], /):

    def wrapper[**P, T](func: cabc.Callable[P, T]) -> cabc.Callable[P, T]:
        def function(*args: P.args, **kwargs: P.kwargs) -> T:
            with re_raise_scope(before, after):
                return func(*args, **kwargs)

        return function

    return wrapper
