# Copyright (c) AIoWay Authors - All Rights Reserved

"Torch function/dispatch modes, corresponding to `__torch_function__`/`__torch_dispatch__`."

import abc
import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from torch import _ops, overrides
from torch.utils import _python_dispatch as pyd
from aioway._common import find_nested_tensors, render_fcall, replace_tensors
from aioway.schemas import attr

from .fn import Fn
from .guards import TensorFilter, all_tensors

__all__ = [
    "TFunctionMode",
    "TDispatchMode",
    "TFunctionFn",
    "TDispatchFn",
    "HasParam",
]


type _TorchCallable = cabc.Callable[..., typing.Any] | _ops.OpOverload


class HasParam(abc.ABC):
    """
    `HasParam` is a mixin that requires you to implement `tensors`,
    providing `parameters(select)` which iterates over the tensors and filter them.
    """

    def parameters(self, select: TensorFilter = all_tensors, /):
        """
        Calls `.tensors()` and then use `select` to iterate over the tensors.
        """

        for tensor in self.tensors():
            if select(tensor):
                yield tensor

    @abc.abstractmethod
    def tensors(self) -> cabc.Iterator[torch.Tensor]:
        """
        All the tensors that this `Fn` uses.
        """

        raise NotImplementedError


@dcls.dataclass(match_args=False)
class _TThunkBaseFn[T: _TorchCallable](HasParam, Fn, abc.ABC):
    """
    `TorchThunkFn` is the thunk capturing the function calls initiated by `torch`.
    It's the base class for both `TorchFunctionFn` and `TorchDispatchFn`
    """

    __match_args__ = "func", "types", "args", "kwargs"

    func: T
    "The `torch.*`, `Tensor.*` functions."

    types: tuple[type, ...]
    "The types of the arguments."

    args: tuple[typing.Any, ...]
    "The positional args."

    kwargs: dict[str, typing.Any]
    "The keyword arguments."

    def __post_init__(self):
        if not callable(self.func):
            raise TypeError(f"{self.func=} is not callable.")

        if not isinstance(self.args, tuple):
            raise TypeError(f"{self.args=} is not a tuple.")

        if not isinstance(self.kwargs, dict):
            raise TypeError(f"{self.kwargs=} is not a dict.")

    def __iter__(self) -> cabc.Iterator[typing.Any]:
        yield self.func
        yield self.types
        yield self.args
        yield self.kwargs

    @typing.override
    @typing.no_type_check
    def do(self) -> torch.Tensor:
        return self.func(*self.args, **self.kwargs)

    @typing.override
    def tensors(self) -> cabc.Iterator[torch.Tensor]:
        yield from find_nested_tensors(self.args)
        yield from find_nested_tensors(self.kwargs)


@dcls.dataclass(match_args=False)
class TFunctionFn(_TThunkBaseFn[cabc.Callable[..., typing.Any]]):
    """
    `TorchFunctionT` is the thunk capturing the function calls initiated by `torch`.

    The `func` here are `torch.*` or `torch.Tensor` operators.
    """

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        func_name = _render_func_name(self.func)
        return _render_replace_tensors("function::" + func_name, self.args, self.kwargs)


@dcls.dataclass(match_args=False)
class TDispatchFn(_TThunkBaseFn[_ops.OpOverload]):
    """
    `TorchDispatchT` is the thunk capturing the function calls initiated by `torch`.
    This is by default what a null-op `__torch_dispatch__` would call.

    The `func` here are `torch.ops.aten.*` operators.
    """

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.func, _ops.OpOverload):
            raise TypeError(f"{self.func=} is not a `torch._ops.OpOverload`.")

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        func_name = _render_func_name(self.func)
        return _render_replace_tensors("dispatch::" + func_name, self.args, self.kwargs)


def _render_func_name(func: cabc.Callable[..., typing.Any]) -> str:
    name = func.__name__

    # If it's `torch.*`.
    if getattr(torch, name, None) is func:
        return f"torch.{name}"

    # If it's `torch.Tensor.*`.
    if getattr(torch.Tensor, name, None) is func:
        return f"torch.Tensor.{name}"

    if isinstance(func, _ops.OpOverload):
        return f"torch.ops.{func.namespace}.{func.__name__}"

    # For torchvision items, a `torch._ops.OpOverloadPacket` is passed.
    if isinstance(func, _ops.OpOverloadPacket):
        return f"torch.ops.{func!s}"

    # Don't know what this is. Use `repr`.
    return repr(func)


@typing.no_type_check
def _render_replace_tensors(
    func: str, args: tuple[typing.Any, ...], kwargs: dict[str, typing.Any]
):
    # `Attr`s are better for display than `torch.Tensor`s.
    args = replace_tensors(args, attr)
    kwargs = replace_tensors(kwargs, attr)
    return render_fcall(func, *args, **kwargs)


class TMode[T](typing.Protocol):
    @abc.abstractmethod
    def __call__(self, thunk: T, /) -> torch.Tensor:
        raise NotImplementedError


class TFunctionMode(overrides.TorchFunctionMode, TMode[TFunctionFn], abc.ABC):
    """
    `TorchFunctionMode` is the adaptor for `torch.overrides.TorchFunctionMode`.
    """

    @abc.abstractmethod
    def __call__(self, thunk: TFunctionFn, /) -> torch.Tensor:
        raise NotImplementedError

    @typing.final
    @typing.override
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        thunk = TFunctionFn(func, types, args, kwargs)
        return self(thunk)

    @staticmethod
    def register(
        f: TMode[TFunctionFn],
    ) -> cabc.Callable[[], typing.ContextManager[None]]:

        class _FuncTorchFunctionMode(TFunctionMode):
            @typing.override
            def __call__(self, t: TFunctionFn) -> typing.Any:
                return f(t)

        @ctxl.contextmanager
        def ctx_man():
            with _FuncTorchFunctionMode():
                yield

        return ctx_man


class TDispatchMode(pyd.TorchDispatchMode, TMode[TDispatchFn], abc.ABC):
    """
    `TorchDispatchMode` is the adaptor for `torch.data._python_dispatch.TorchDispatchMode`.
    """

    @abc.abstractmethod
    def __call__(self, thunk: TDispatchFn, /) -> torch.Tensor:
        raise NotImplementedError

    @typing.final
    @typing.override
    def __torch_dispatch__(self, func, types, args=(), kwargs=None) -> typing.Any:
        kwargs = kwargs or {}
        thunk = TDispatchFn(func, types, args, kwargs)
        return self(thunk)

    @staticmethod
    def register(
        f: TMode[TDispatchFn],
    ) -> cabc.Callable[[], typing.ContextManager[None]]:
        class _FuncTorchDispatchMode(TDispatchMode):
            @typing.override
            def __call__(self, t: TDispatchFn, /) -> torch.Tensor:
                return f(t)

        @ctxl.contextmanager
        def ctx_man():
            with _FuncTorchDispatchMode():
                yield

        return ctx_man
