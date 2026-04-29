# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import dataclasses as dcls
import functools
import logging
import typing
from collections import abc as cabc

import torch
from torch import _ops
from torch.utils import _python_dispatch as pyd

from aioway._common.tracking.logging import enable_rich_log
from aioway.ctx import enabled_fake_mode, fake_mode
from aioway.fn.funcs import track_function_fn
from aioway.schemas.attrs import attr

from .fn import Fn, FnStack, Thunk
from .guards import TensorFilter, all_tensors, is_aten_op, is_leaf_has_grad, is_prim_op
from .previews import Preview, TorchDispatchThunk, TorchFn, find_preview

__all__ = [
    "print_torch_dispatch",
    "log_torch_dispatch",
    "log_and_enable_rich",
    "track_dispatch_fn",
    "fake_dispatch_fn",
    "FnList",
    "TrackDispatchMode",
]

LOGGER = logging.getLogger(__name__)


_CreatePreview = cabc.Callable[..., Preview]
_TorchRouterMode = typing.Literal["dispatch", "function"]

_DISPATCH_STACK = FnStack[TorchFn]()


@typing.runtime_checkable
class TorchRouter(typing.Protocol):
    def __call__(
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        args: tuple[typing.Any, ...],
        kwargs: dict[str, typing.Any],
    ) -> torch.Tensor: ...


class TorchRouterFactory(typing.Protocol):
    def __call__(
        self, func: _ops.OpOverload, types: tuple[type[torch.Tensor], ...]
    ) -> _CreatePreview: ...


class _PrintDispatch(pyd.TorchDispatchMode):
    def __torch_dispatch__(
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        args: tuple[typing.Any, ...] = (),
        kwargs: dict[str, typing.Any] | None = None,
    ):
        kwargs = kwargs or {}
        invoke = Thunk(func, *args, **kwargs)
        return self.invoke_and_print(invoke)

    def invoke_and_print(self, invoke: Thunk):
        result = invoke.do()
        print(f"{invoke!s} -> {result!r}")
        return result


print_torch_dispatch = _PrintDispatch
"""
Print the dispatcher.
"""


@dcls.dataclass
class _LogDispatch(_PrintDispatch):
    """
    Log every call to dispatch mode.
    """

    level: int
    "The level to log to."

    logger: logging.Logger = LOGGER
    "The logger to log to. Default to the one in the current module."

    @typing.override
    def invoke_and_print(self, invoke: Thunk):
        result = invoke.do()
        self.logger.log(self.level, "%s -> %s", invoke, result)
        return result


log_torch_dispatch = _LogDispatch
"""
Context manager to log the `__torch_dispatch__` calls.

Args:
    logger: The logger to use. Default to the one in this module.
    level: The level to log to. Default to `logging.DEBUG`.
"""


@ctxl.contextmanager
def log_and_enable_rich(level: int, /):
    with enable_rich_log(level) as logger, log_torch_dispatch(level):
        yield logger


def only_route_aten_in_fake(
    func: _ops.OpOverload, types: tuple[type[torch.Tensor], ...]
) -> _CreatePreview:
    if not enabled_fake_mode():
        raise RuntimeError("Only running in fake mode!")

    if is_aten_op(func):
        return aten_ops_preview(func, types)

    assert is_prim_op(func), func
    return NotImplemented


def no_route(*args, **kwargs) -> _CreatePreview:
    return NotImplemented


@dcls.dataclass(frozen=True)
class FnList:
    """
    The list of `TorchFn` that tracks the current history.
    """

    history: list[TorchFn] = dcls.field(default_factory=list)
    """
    The `TorchFn` that has been called, in order.
    """

    fn_index: dict[torch.Tensor, TorchFn] = dcls.field(default_factory=dict)
    "The mapping from output to tensor input."

    def __len__(self) -> int:
        return len(self.history)

    def __getitem__(self, idx: int) -> Fn:
        return self.history[idx]

    def __iter__(self):
        yield from self.history

    def append(self, item: TorchFn, /):
        self.history.append(item)

    def pop(self):
        return self.history.pop()

    def parameters(self, select: TensorFilter = is_leaf_has_grad, unique: bool = True):
        def data_params():
            for fn in self.history:
                yield from fn.parameters(select)

        params = data_params()

        if unique:
            params = set(data_params())

        yield from params

    def numel(self) -> int:
        return sum(param.numel() for param in self.parameters(all_tensors))

    def memory(self) -> int:
        return sum(attr(param).memory() for param in self.parameters(all_tensors))

    def find_fn(self, tensor: torch.Tensor):
        return self.fn_index[tensor]


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
            result = fn.do()

        # Store it in the history.
        self.history.fn_index[result] = fn
        return result


def aten_ops_preview(
    func: _ops.OpOverload, types: tuple[type[torch.Tensor], ...]
) -> _CreatePreview:
    assert is_aten_op(func), func
    return functools.partial(find_preview, func)


@ctxl.contextmanager
def capture_do_error(fn: TorchFn):
    try:
        yield
    except RuntimeError as err:
        raise ValueError(f"Function call '{fn}' failed.") from err


@ctxl.contextmanager
def track_dispatch_fn(router: TorchRouterFactory = no_route):
    """
    Track all calls into the torch dispatch mode as `TorchIrFn`.
    """

    with track_function_fn(), TrackDispatchMode(router=router) as sdm:
        yield sdm.history


@ctxl.contextmanager
def fake_dispatch_fn():
    """
    Track all calls into the torch dispatch mode as `TorchIrFn`,
    when fake mode is active.
    """

    with fake_mode(), track_dispatch_fn(only_route_aten_in_fake) as history:
        yield history


def torch_dispatch_stack():
    return _DISPATCH_STACK
