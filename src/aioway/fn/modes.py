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
from aioway._common.torch import find_nested_tensors

from .fn import Fn
from .previews import TensorFn

__all__ = [
    "TorchFunctionMode",
    "TorchDispatchMode",
    "TorchFunctionFn",
    "TorchDispatchFn",
]


class _TorchLikeFunc(typing.Protocol):
    """
    The API that torch uses for their custom dispatchers.

    This is the protocol that constrains our implementations to follow the same signature.
    """

    def __call__(self, func, types, *args, **kwargs) -> typing.Any: ...


type _TorchCallable = cabc.Callable[..., typing.Any] | _ops.OpOverload


@dcls_no_repr
class TorchThunkFn[T: _TorchCallable](TensorFn, abc.ABC):
    """
    `TorchThunkFn` is the thunk capturing the function calls initiated by `torch`.
    It's the base class for both `TorchFunctionFn` and `TorchDispatchFn`
    """

    func: T
    "The `torch.*`, `Tensor.*` functions."

    types: tuple[type, ...]
    "The types of the arguments."

    args: tuple[typing.Any, ...]
    "The positional args."

    kwargs: dict[str, typing.Any]
    "The keyword arguments."

    @typing.override
    @typing.no_type_check
    def do(self) -> torch.Tensor:
        return self.func(*self.args, **self.kwargs)

    @typing.override
    def tensors(self) -> cabc.Iterator[torch.Tensor]:
        yield from find_nested_tensors(self.args)
        yield from find_nested_tensors(self.kwargs)


@dcls_no_repr
class TorchFunctionFn(TorchThunkFn[cabc.Callable[..., typing.Any]], Fn):
    """
    `TorchFunctionT` is the thunk capturing the function calls initiated by `torch`.
    """

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return render_fcall(self.func, *self.args, **self.kwargs)


@dcls_no_repr
class TorchDispatchFn(TorchThunkFn[_ops.OpOverload]):
    """
    `TorchDispatchT` is the thunk capturing the function calls initiated by `torch`.
    This is by default what a null-op `__torch_dispatch__` would call.
    """

    func: _ops.OpOverload
    "The `torch.ops.*` operator."

    types: tuple[type, ...]
    "The types of the arguments."

    args: tuple[typing.Any, ...]
    "The positional args."

    kwargs: dict[str, typing.Any]
    "The keyword arguments."

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return render_fcall(self.func.name(), *self.args, **self.kwargs)


type _FunctionOrDispatch = TorchFunctionFn | TorchDispatchFn


class TorchMode[T: TorchThunkFn](abc.ABC):
    @abc.abstractmethod
    def __call__(self, thunk: T, /) -> torch.Tensor:
        raise NotImplementedError


class TorchFunctionMode(
    overrides.TorchFunctionMode, TorchMode[TorchFunctionFn], abc.ABC
):
    """
    `TorchFunctionMode` is the adaptor for `torch.overrides.TorchFunctionMode`.

    It automatically stores the current active mode into a stack,
    so we have better observing and debugging ability.
    """

    @abc.abstractmethod
    def __call__(self, thunk: TorchFunctionFn, /) -> torch.Tensor:
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
    def register(f: _TorchLikeFunc) -> cabc.Callable[[], typing.ContextManager[None]]:

        class _FuncTorchFunctionMode(TorchFunctionMode):
            @typing.override
            def __call__(self, t: TorchFunctionFn) -> typing.Any:
                return f(t.func, t.types, *t.args, **t.kwargs)

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
    def __call__(self, thunk: TorchDispatchFn, /) -> torch.Tensor:
        raise NotImplementedError

    @typing.final
    @typing.override
    def __torch_dispatch__(self, op, types, args=..., kwargs=None) -> typing.Any:
        kwargs = kwargs or {}
        args = () if args == ... else args

        thunk = TorchDispatchFn(func=op, types=types, args=args, kwargs=kwargs)
        with active_dispatch_modes().track(thunk):
            result = thunk.do()
        return result

    @staticmethod
    def register(f: _TorchLikeFunc) -> cabc.Callable[[], typing.ContextManager[None]]:
        class _FuncTorchDispatchMode(TorchDispatchMode):
            @typing.override
            def __call__(self, thunk: TorchDispatchFn, /) -> torch.Tensor:
                return f(t.func, t.types, *t.args, **t.kwargs)

        @ctxl.contextmanager
        def ctx_man():
            with _FuncTorchDispatchMode():
                yield

        return ctx_man
