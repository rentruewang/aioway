# Copyright (c) AIoWay Authors - All Rights Reserved

"Torch functions, corresponding to `__torch_function__` mode."

import abc
import contextlib as ctxl
import typing
from collections import abc as cabc

import torch
from torch import _ops, overrides
from torch.utils import _python_dispatch as pyd

from aioway._common import dcls_no_repr, render_fcall

from .fn import Fn
from .previews import HasParams

__all__ = [
    "TorchFunctionMode",
    "TorchDispatchMode",
    "active_function_modes",
    "active_dispatch_modes",
]

_ACTIVE_FUNCTION_MODES: list[_TorchFunctionModeT] = []
_ACTIVE_DISPATCH_MODES: list[_TorchDispatchModeT] = []


class _TorchMode(typing.Protocol):
    """
    The API that torch uses for their custom dispatchers.

    This is the protocol that constrains our implementations to follow the same signature.
    """

    def __call__(
        self,
        func: cabc.Callable[..., typing.Any],
        types: tuple[type[torch.Tensor], ...],
        args: tuple[typing.Any, ...],
        kwargs: dict[str, typing.Any],
    ) -> typing.Any: ...


class TorchFunctionMode(overrides.TorchFunctionMode, abc.ABC):
    """
    `TorchFunctionMode` is the adaptor for `torch.overrides.TorchFunctionMode`.

    It automatically stores the current active mode into a stack,
    so we have better observing and debugging ability.
    """

    @abc.abstractmethod
    def __call__(
        self,
        func: cabc.Callable[..., typing.Any],
        types: tuple[type, ...],
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        raise NotImplementedError

    @typing.final
    @typing.override
    def __torch_function__(self, func, types, args=..., kwargs=None) -> typing.Any:
        kwargs = kwargs or {}
        args = () if args == ... else args

        thunk = _TorchFunctionModeT(self, func, types, args, kwargs)
        with _push_current_call(thunk, _ACTIVE_FUNCTION_MODES):
            return self(func, types, *args, **kwargs)

    @staticmethod
    def register(func: _TorchMode) -> cabc.Callable[[], typing.ContextManager[None]]:

        class _FuncTorchFunctionMode(TorchFunctionMode):
            @typing.override
            def __call__(self, *args, **kwargs):
                return func(*args, **kwargs)

        @ctxl.contextmanager
        def ctx_man():
            with _FuncTorchFunctionMode():
                yield

        return ctx_man


class TorchDispatchMode(pyd.TorchDispatchMode, abc.ABC):
    """
    `TorchDispatchMode` is the adaptor for `torch.data._python_dispatch.TorchDispatchMode`.

    It automatically stores the current active mode into a stack,
    so we have better observing and debugging ability.
    """

    @abc.abstractmethod
    def __call__(
        self,
        op: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> torch.Tensor:
        raise NotImplementedError

    @typing.final
    @typing.override
    def __torch_dispatch__(self, op, types, args=..., kwargs=None) -> typing.Any:
        kwargs = kwargs or {}
        args = () if args == ... else args

        thunk = _TorchDispatchModeT(self, op, types, args, kwargs)
        with _push_current_call(thunk, _ACTIVE_DISPATCH_MODES):
            result = self(op, types, *args, **kwargs)
        return result

    @staticmethod
    def register(func: _TorchMode) -> cabc.Callable[[], typing.ContextManager[None]]:
        class _FuncTorchDispatchMode(TorchDispatchMode):
            def __call__(self, *args, **kwargs):
                return func(*args, **kwargs)

        @ctxl.contextmanager
        def ctx_man():
            with _FuncTorchDispatchMode():
                yield

        return ctx_man


@dcls_no_repr
class TorchFunctionT:
    func: cabc.Callable[..., typing.Any]
    types: tuple[type, ...]
    args: tuple[typing.Any, ...]
    kwargs: dict[str, typing.Any]

    def __repr__(self) -> str:
        return render_fcall(self.func, *self.args, **self.kwargs)


@dcls_no_repr
class TorchDispatchT(HasParams, Fn):
    op: _ops.OpOverload
    types: tuple[type, ...]
    args: tuple[typing.Any, ...]
    kwargs: dict[str, typing.Any]

    @typing.override
    def do(self):
        return self.op(*self.args, **self.kwargs)

    @typing.override
    def tensors(self) -> cabc.Iterator[torch.Tensor]:
        for arg in self.args:
            if isinstance(arg, torch.Tensor):
                yield arg

        for arg in self.kwargs.values():
            if isinstance(arg, torch.Tensor):
                yield arg

    def __repr__(self) -> str:
        return render_fcall(self.op.name(), *self.args, **self.kwargs)


@dcls_no_repr
class _HasMode[T]:
    mode: T


@dcls_no_repr
class _TorchFunctionModeT(TorchFunctionT, _HasMode[TorchFunctionMode]): ...


@dcls_no_repr
class _TorchDispatchModeT(TorchDispatchT, _HasMode[TorchDispatchMode]): ...


@ctxl.contextmanager
def _push_current_call[T](item: T, stack: list[T]):
    """
    Add the current call to the stack, pop after.
    """

    try:
        stack.append(item)
        yield
    finally:
        _ = stack.pop()


def active_function_modes():
    return _ACTIVE_FUNCTION_MODES


def active_dispatch_modes():
    return _ACTIVE_DISPATCH_MODES


def active_functions():
    """
    Get all the currently active torch functions.
    """

    yield from {mode.func for mode in active_function_modes()}


def active_dispatches():
    """
    Get all the current active torch dispatches.
    """

    yield from {mode.op for mode in active_dispatch_modes()}
