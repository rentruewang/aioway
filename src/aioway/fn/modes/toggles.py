# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import typing

from ._on_off import OnOffStack
from .modules import FORWARDS, INITS
from .tensors import DISPATCHES, FUNCTIONS

__all__ = ["active_mode", "set_mode", "mode_off"]

type _ModeName = typing.Literal["function", "dispatch", "init", "forward"]


def active_mode(mode: _ModeName):
    """
    Get the mode for
    """
    match mode:
        case "function":
            return FUNCTIONS
        case "dispatch":
            return DISPATCHES
        case "init":
            return INITS
        case "forward":
            return FORWARDS
        case _:
            raise ValueError(f"Unexpected {mode=} is not supported.")


@ctxl.contextmanager
def set_mode(
    *,
    function: bool | None = None,
    dispatch: bool | None = None,
    init: bool | None = None,
    forward: bool | None,
):
    """
    Turn on or off `__torch_function__` / `__torch_dispatch__` mode for the given scope,
    for the modes that are **currently activated**.

    If not provided, the modes won't be set is unchanged.

    Args:
        function: Enable the `__torch_function__` mode.
        dispatch: Enable the `__torch_dispatch__` mode.
        init: Enable the hooks during `nn.Module.__init__`.
        forward: Enable the hooks during `nn.Module.forward`.

    Note:
        We are implementing this flag instead of using `no_dispatch` utility from `torch`,
        is because thier version causes segmentation fault in some cases.
    """

    with ctxl.ExitStack() as ctx:

        def _switch_if_not_none(stack: OnOffStack, flag: bool | None):
            if flag is not None:
                ctx.enter_context(stack.switch(flag))

        _switch_if_not_none(FUNCTIONS, function)
        _switch_if_not_none(DISPATCHES, dispatch)
        _switch_if_not_none(INITS, init)
        _switch_if_not_none(FORWARDS, forward)

        yield


@ctxl.contextmanager
def mode_off():
    with set_mode(function=False, dispatch=False, init=False, forward=False):
        yield
