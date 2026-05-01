# Copyright (c) AIoWay Authors - All Rights Reserved

"Torch functions, corresponding to `__torch_function__` mode."

import contextlib as ctxl
import dataclasses as dcls
import typing

import torch
from torch import _ops, overrides
from torch.utils import _python_dispatch as pyd

from aioway.fn import FnStack, Thunk, TorchDispatchThunk, TorchFn

__all__ = ["function_fn_stack", "torch_function_stack"]

_FUNCTION_STACK = FnStack[Thunk]()
_DISPATCH_STACK = FnStack[TorchFn]()


@dcls.dataclass(frozen=True)
class _FunctionStack(overrides.TorchFunctionMode):
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
class _DispatchStack(pyd.TorchDispatchMode):

    def __torch_dispatch__(
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        args: tuple[typing.Any, ...] = (),
        kwargs: dict[str, typing.Any] | None = None,
    ):
        kwargs = kwargs or {}
        fn = TorchDispatchThunk(func, types, *args, **kwargs)
        with _DISPATCH_STACK.track(fn):
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

    # Create a new one, but rememeber to reset it once done.
    with _FunctionStack() as _function_tracker:
        yield _function_tracker


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

    return _FUNCTION_STACK
