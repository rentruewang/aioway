# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import typing


class FnCtx[T](typing.Protocol):
    """
    A `FnCtx` is just a context manager.
    """

    def __call__(self, *args, **kwargs) -> ctxl.AbstractContextManager[T]: ...


_CONTEXTS: set[FnCtx] = set()
