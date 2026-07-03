# Copyright (c) AIoWay Authors - All Rights Reserved

from collections import abc as cabc

__all__ = ["re_raise"]


def re_raise(before: type[BaseException], after: type[BaseException], /):

    def wrapper[**P, T](func: cabc.Callable[P, T]) -> cabc.Callable[P, T]:
        def function(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except before as exc:
                raise after from exc

        return function

    return wrapper
