# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import dataclasses as dcls
import logging
import typing
from collections import abc as cabc

import torch
from torch import _ops

from aioway.schemas import attr

from .fake import enabled_fake_mode, fake_mode
from .guards import TensorFilter, all_tensors, is_aten_op, is_leaf_has_grad, is_prim_op
from .modes import TorchDispatchFn, TorchDispatchMode
from .previews import PreviewFn, PreviewFnFinder

__all__ = [
    "track_dispatch_fn",
    "fake_dispatch_fn",
    "FnList",
    "RouteDispatchOp",
]

LOGGER = logging.getLogger(__name__)


_CreatePreview = cabc.Callable[..., PreviewFn]
_TorchRouterMode = typing.Literal["dispatch", "function"]
_TorchThunk = TorchDispatchFn | PreviewFn


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
    The list of `_TorchOp` that tracks the current history.
    """

    history: list[_TorchThunk] = dcls.field(default_factory=list)
    """
    The `TorchFn` that has been called, in order.
    """

    fn_index: dict[torch.Tensor, _TorchThunk] = dcls.field(default_factory=dict)
    "The mapping from output to tensor input."

    def __len__(self) -> int:
        return len(self.history)

    def __getitem__(self, idx: int) -> _TorchThunk:
        return self.history[idx]

    def __iter__(self):
        yield from self.history

    def append(self, item: _TorchThunk, /):
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
class RouteDispatchOp(TorchDispatchMode):
    router: TorchRouterFactory
    history: FnList = dcls.field(default_factory=FnList)

    def __call__(
        self,
        op: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        *args: tuple[typing.Any, ...],
        **kwargs: dict[str, typing.Any],
    ):
        # Create a `_ThunkType` and route implemented methods.
        fn_init = self.router(op, types)
        fn: _TorchThunk

        if (
            False
            # Not ATen operator.
            or fn_init is NotImplemented
            # Fn is not handled.
            or (fn := fn_init(*args, **kwargs)) is NotImplemented
        ):
            fn = TorchDispatchFn(op, types, args, kwargs)

        assert isinstance(fn, _TorchThunk), type(fn)
        self.history.append(fn)

        # Here, we overwrite `fn`'s `__call__` inside `PreviewFn` if it's a special function.
        with capture_do_error(fn):
            result = fn.do()

        # Store it in the history.
        self.history.fn_index[result] = fn
        return result


def aten_ops_preview(
    func: _ops.OpOverload, types: tuple[type[torch.Tensor], ...]
) -> _CreatePreview:
    assert is_aten_op(func), func
    return PreviewFnFinder(func)


@ctxl.contextmanager
def capture_do_error(fn: _TorchThunk):
    try:
        yield
    except RuntimeError as err:
        raise ValueError(f"Function call '{fn}' failed.") from err


@ctxl.contextmanager
def track_dispatch_fn(router: TorchRouterFactory = no_route):
    """
    Track all calls into the torch dispatch mode as `TorchIrFn`.
    """

    with RouteDispatchOp(router=router) as sdm:
        yield sdm.history


@ctxl.contextmanager
def fake_dispatch_fn():
    """
    Track all calls into the torch dispatch mode as `TorchIrFn`,
    when fake mode is active.
    """

    with fake_mode(), track_dispatch_fn(only_route_aten_in_fake) as history:
        yield history
