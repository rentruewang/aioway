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

from aioway._common import render_fcall
from aioway._common.torch import find_nested_tensors

from .fn import Fn
from .guards import TensorFilter, all_tensors

__all__ = [
    "TorchFunctionMode",
    "TorchDispatchMode",
    "TorchFunctionFn",
    "TorchDispatchFn",
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
class _TorchThunkBaseFn[T: _TorchCallable](HasParam, Fn, abc.ABC):
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
class TorchFunctionFn(_TorchThunkBaseFn[cabc.Callable[..., typing.Any]], Fn):
    """
    `TorchFunctionT` is the thunk capturing the function calls initiated by `torch`.
    """

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return render_fcall(self.func, *self.args, **self.kwargs)


@dcls.dataclass(match_args=False)
class TorchDispatchFn(_TorchThunkBaseFn[_ops.OpOverload]):
    """
    `TorchDispatchT` is the thunk capturing the function calls initiated by `torch`.
    This is by default what a null-op `__torch_dispatch__` would call.
    """

    func: _ops.OpOverload
    "The `torch.ops.*` operator."

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.func, _ops.OpOverload):
            raise TypeError(f"{self.func=} is not a `torch._ops.OpOverload`.")

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return render_fcall(self.func.name(), *self.args, **self.kwargs)


type _FunctionOrDispatch = TorchFunctionFn | TorchDispatchFn


class TorchMode[T](typing.Protocol):
    @abc.abstractmethod
    def __call__(self, thunk: T, /) -> torch.Tensor:
        raise NotImplementedError


class TorchFunctionMode(
    overrides.TorchFunctionMode, TorchMode[TorchFunctionFn], abc.ABC
):
    """
    `TorchFunctionMode` is the adaptor for `torch.overrides.TorchFunctionMode`.
    """

    @abc.abstractmethod
    def __call__(self, thunk: TorchFunctionFn, /) -> torch.Tensor:
        raise NotImplementedError

    @typing.final
    @typing.override
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        thunk = TorchFunctionFn(func, types, args, kwargs)
        return self(thunk)

    @staticmethod
    def register(
        f: TorchMode[TorchFunctionFn],
    ) -> cabc.Callable[[], typing.ContextManager[None]]:

        class _FuncTorchFunctionMode(TorchFunctionMode):
            @typing.override
            def __call__(self, t: TorchFunctionFn) -> typing.Any:
                return f(t)

        @ctxl.contextmanager
        def ctx_man():
            with _FuncTorchFunctionMode():
                yield

        return ctx_man


class TorchDispatchMode(pyd.TorchDispatchMode, TorchMode[TorchDispatchFn], abc.ABC):
    """
    `TorchDispatchMode` is the adaptor for `torch.data._python_dispatch.TorchDispatchMode`.
    """

    @abc.abstractmethod
    def __call__(self, thunk: TorchDispatchFn, /) -> torch.Tensor:
        raise NotImplementedError

    @typing.final
    @typing.override
    def __torch_dispatch__(self, func, types, args=(), kwargs=None) -> typing.Any:
        kwargs = kwargs or {}
        thunk = TorchDispatchFn(func, types, args, kwargs)
        return self(thunk)

    @staticmethod
    def register(
        f: TorchMode[TorchDispatchFn],
    ) -> cabc.Callable[[], typing.ContextManager[None]]:
        class _FuncTorchDispatchMode(TorchDispatchMode):
            @typing.override
            def __call__(self, t: TorchDispatchFn, /) -> torch.Tensor:
                return f(t)

        @ctxl.contextmanager
        def ctx_man():
            with _FuncTorchDispatchMode():
                yield

        return ctx_man
