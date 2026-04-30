# Copyright (c) AIoWay Authors - All Rights Reserved

"Torch functions, corresponding to `__torch_function__` mode."

import contextlib as ctxl
import dataclasses as dcls
import typing

import torch
from torch import _ops, overrides

from .fn import FnStack, Thunk

__all__ = ["function_fn_stack"]

_function_tracker: _TrackFunctionMode | None = None
"The current function tracker. It's a singleton."


@dcls.dataclass(frozen=True)
class _TrackFunctionMode(overrides.TorchFunctionMode):
    functions: list[Thunk] = dcls.field(default_factory=list)
    stack: FnStack[Thunk] = dcls.field(default_factory=FnStack)

    @typing.override
    def __torch_function__(
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        args: tuple[typing.Any, ...] = (),
        kwargs: dict[str, typing.Any] | None = None,
    ) -> typing.Any:
        kwargs = kwargs or {}
        fn = Thunk(func, *args, **kwargs)
        self.functions.append(fn)

        with self.stack.track(fn):
            return func(*args, **kwargs)


@dcls.dataclass
class TrackDispatchMode(pyd.TorchDispatchMode):
    router: TorchRouterFactory
    history: FnList = dcls.field(default_factory=FnList)

    def __torch_dispatch__(
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        args: tuple[typing.Any, ...] = (),
        kwargs: dict[str, typing.Any] | None = None,
    ):
        kwargs = kwargs or {}

        # Create a `_ThunkType` and route implemented methods.
        fn_init = self.router(func, types)
        fn: TorchFn

        if (
            False
            # Not ATen operator.
            or fn_init is NotImplemented
            # Fn is not handled.
            or (fn := fn_init(*args, **kwargs)) is NotImplemented
        ):
            fn = TorchDispatchThunk(func, types, *args, **kwargs)

        assert isinstance(fn, TorchFn), fn
        self.history.append(fn)

        with _DISPATCH_STACK.track(fn), capture_do_error(fn):
            result = func(*args, **kwargs)

        # Store it in the history.
        self.history.fn_index[result] = fn
        return result


@ctxl.contextmanager
def function_fn_stack():
    """
    Activate the tracker that will track all `torch.*` calls.
    If a tracker is created, it will be reused.
    """

    global _function_tracker

    # If it already exists, don't create a new one.
    if _function_tracker:
        yield _function_tracker.functions
        return

    # Create a new one, but rememeber to reset it once done.
    with _TrackFunctionMode() as _function_tracker:
        try:
            yield _function_tracker.functions
        finally:
            _function_tracker = None


def function_tracker():
    """
    Retrieve the current function tracker that is active.
    Raise `RuntimeError` if one is not found.
    """

    if _function_tracker is None:
        raise RuntimeError("The function tracker is not active yet.")

    return _function_tracker


def torch_function_stack():
    "Get the `__torch_function__` stack that is used when `track_function_fn` is enabled."

    return function_tracker().stack
