# Copyright (c) AIoWay Authors - All Rights Reserved

"Torch functions, corresponding to `__torch_function__` mode."

import abc
import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from torch import _ops, overrides
from torch.utils import _python_dispatch as pyd

from .fn import Thunk

__all__ = ["TorchFunctionMode", "TorchDispatchMode"]

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
    def register(
        func: _TorchMode,
    ) -> cabc.Callable[[], typing.ContextManager[None]]:
        @ctxl.contextmanager
        def ctx_man():
            with _FuncFunMode(func):
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
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> torch.Tensor:
        raise NotImplementedError

    @typing.final
    @typing.override
    def __torch_dispatch__(self, func, types, args=..., kwargs=None) -> typing.Any:
        kwargs = kwargs or {}
        args = () if args == ... else args

        thunk = _TorchDispatchModeT(self, func, types, args, kwargs)
        with _push_current_call(thunk, _ACTIVE_DISPATCH_MODES):
            result = self(func, types, *args, **kwargs)
        assert isinstance(result, torch.Tensor)
        return result

    @staticmethod
    def register(
        func: _TorchMode,
    ) -> cabc.Callable[[], typing.ContextManager[None]]:
        @ctxl.contextmanager
        def ctx_man():
            with _FuncDisMode(func):
                yield

        return ctx_man


@dcls.dataclass(frozen=True)
class _TorchModeLike:
    func: typing.Any
    types: tuple[type, ...]
    args: tuple[typing.Any, ...]
    kwargs: dict[str, typing.Any]

    @typing.override
    def __repr__(self) -> str:
        return repr(Thunk(self.func, *self.args, **self.kwargs))


class TorchFunctionT(_TorchModeLike):
    func: cabc.Callable[..., typing.Any]


@dcls.dataclass(frozen=True)
class TorchDispatchT(_TorchModeLike):
    func: _ops.OpOverload


@dcls.dataclass(frozen=True)
class _HasMode[T]:
    mode: T


@dcls.dataclass(frozen=True)
class _TorchFunctionModeT(TorchFunctionT, _HasMode[TorchFunctionMode]): ...


@dcls.dataclass(frozen=True)
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
        stack.pop()


@dcls.dataclass(frozen=True)
class _FuncFunMode(TorchFunctionMode):
    func: _TorchMode

    @typing.override
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


@dcls.dataclass(frozen=True)
class _FuncDisMode(TorchDispatchMode):
    func: _TorchMode

    @typing.override
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
