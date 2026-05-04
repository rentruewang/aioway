# Copyright (c) AIoWay Authors - All Rights Reserved

"Torch function/dispatch modes, corresponding to `__torch_function__`/`__torch_dispatch__`."

import abc
import contextlib as ctxl
import typing
from collections import abc as cabc

import torch
from torch import _ops, overrides
from torch.utils import _python_dispatch as pyd

from aioway._common import dcls_no_repr, render_fcall

from .fn import Fn, FnStack
from .previews import TensorFn

__all__ = [
    "TorchFunctionMode",
    "TorchDispatchMode",
    "active_function_modes",
    "active_dispatch_modes",
    "TorchFunctionFn",
    "TorchDispatchFn",
    "TorchFunctionModeFn",
    "TorchDispatchModeFn",
]

_ACTIVE_FUNCTION_MODES: FnStack[TorchFunctionModeFn] = FnStack()
_ACTIVE_DISPATCH_MODES: FnStack[TorchDispatchModeFn] = FnStack()


class _TorchMode(typing.Protocol):
    """
    The API that torch uses for their custom dispatchers.

    This is the protocol that constrains our implementations to follow the same signature.
    """

    def __call__(self, func, types, *args, **kwargs) -> typing.Any: ...


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

        thunk = TorchFunctionModeFn(self, func, types, args, kwargs)
        with active_function_modes().track(thunk):
            return thunk.do()

    @staticmethod
    def register(f: _TorchMode) -> cabc.Callable[[], typing.ContextManager[None]]:

        class _FuncTorchFunctionMode(TorchFunctionMode):
            @typing.override
            def __call__(
                self,
                func: cabc.Callable[..., typing.Any],
                types: tuple[type, ...],
                *args: typing.Any,
                **kwargs: typing.Any,
            ):
                return f(func, types, *args, **kwargs)

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

        thunk = TorchDispatchModeFn(self, op, types, args, kwargs)
        with active_dispatch_modes().track(thunk):
            result = thunk.do()
        return result

    @staticmethod
    def register(f: _TorchMode) -> cabc.Callable[[], typing.ContextManager[None]]:
        class _FuncTorchDispatchMode(TorchDispatchMode):
            def __call__(
                self,
                op: _ops.OpOverload,
                types: tuple[type[torch.Tensor], ...],
                *args: typing.Any,
                **kwargs: typing.Any,
            ):
                return f(op, types, *args, **kwargs)

        @ctxl.contextmanager
        def ctx_man():
            with _FuncTorchDispatchMode():
                yield

        return ctx_man


@dcls_no_repr
class TorchFunctionFn(Fn):
    """
    `TorchFunctionT` is the thunk capturing the function calls initiated by `torch`.
    """

    func: cabc.Callable[..., typing.Any]
    "The `torch.*`, `Tensor.*` functions."

    types: tuple[type, ...]
    "The types of the arguments."

    args: tuple[typing.Any, ...]
    "The positional args."

    kwargs: dict[str, typing.Any]
    "The keyword arguments."

    def __repr__(self) -> str:
        return render_fcall(self.func, *self.args, **self.kwargs)

    @typing.override
    def do(self) -> torch.Tensor:
        return self.func(*self.args, **self.kwargs)


@dcls_no_repr
class TorchDispatchFn(TensorFn):
    """
    `TorchDispatchT` is the thunk capturing the function calls initiated by `torch`.
    This is by default what a null-op `__torch_dispatch__` would call.
    """

    op: _ops.OpOverload
    "The `torch.ops.*` operator."

    types: tuple[type, ...]
    "The types of the arguments."

    args: tuple[typing.Any, ...]
    "The positional args."

    kwargs: dict[str, typing.Any]
    "The keyword arguments."

    def __hash__(self):
        return id(self)

    def __repr__(self) -> str:
        return render_fcall(self.op.name(), *self.args, **self.kwargs)

    @typing.override
    def do(self) -> torch.Tensor:
        return self.op(*self.args, **self.kwargs)

    @typing.override
    def tensors(self) -> cabc.Iterator[torch.Tensor]:
        for arg in self.args:
            if isinstance(arg, torch.Tensor):
                yield arg

        for arg in self.kwargs.values():
            if isinstance(arg, torch.Tensor):
                yield arg


@dcls_no_repr
class _HasMode[T]:
    mode: T


@dcls_no_repr
class TorchFunctionModeFn(TorchFunctionFn, _HasMode[TorchFunctionMode]):
    """
    `TorchFunctionFn` + `mode` argument.
    Handles passing over the function calls to `mode.__call__`,
    which invokes `TorchFunctionMode` subclasses.
    """

    @typing.override
    def do(self) -> torch.Tensor:
        return self.mode(self.func, self.types, *self.args, **self.kwargs)


@dcls_no_repr
class TorchDispatchModeFn(TorchDispatchFn, _HasMode[TorchDispatchMode]):
    """
    `TorchDispatchFn` + `mode` argument.
    Handles passing over the function calls to `mode.__call__`,
    which invokes `TorchDispatchMode` subclasses.
    """

    @typing.override
    def do(self) -> torch.Tensor:
        return self.mode(self.op, self.types, *self.args, **self.kwargs)


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
