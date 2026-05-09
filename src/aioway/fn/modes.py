# Copyright (c) AIoWay Authors - All Rights Reserved

"Torch function/dispatch modes, corresponding to `__torch_function__`/`__torch_dispatch__`."

import abc
import contextlib as ctxl
import dataclasses as dcls
import types
import typing
from collections import abc as cabc

import torch
from torch import _ops, overrides
from torch.utils import _python_dispatch as pyd

from aioway._common import (
    HasParam,
    find_nested_tensors,
    is_aten_op,
    is_prim_op,
    render_fcall,
)
from aioway._common.breakdowns import NestedFinder
from aioway.fate import Fate, find_fate

from .fn import Fn

__all__ = ["TFunctionMode", "TDispatchMode", "TFunctionFn", "TDispatchFn", "FateFn"]


type _TorchCallable = cabc.Callable[..., typing.Any] | _ops.OpOverload


@dcls.dataclass(match_args=False)
class _TThunkBase[T: _TorchCallable](HasParam, abc.ABC):
    """
    `TorchThunkFn` is the thunk capturing the function calls initiated by `torch`.
    It's the base class for both `TorchFunctionFn` and `TorchDispatchFn`
    """

    __match_args__ = "func", "types", "args", "kwargs"

    _: dcls.KW_ONLY

    func: T
    "The `torch.*`, `Tensor.*` functions."

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

    @typing.override
    @typing.no_type_check
    def do(self) -> torch.Tensor:
        return self.func(*self.args, **self.kwargs)

    @typing.override
    def tensors(self) -> cabc.Iterator[torch.Tensor]:
        yield from find_nested_tensors(self.args)
        yield from find_nested_tensors(self.kwargs)


class TFunctionFnMixin(Fn, abc.ABC):
    "The mixin to define what types `Fn` graph would capture during function mode."

    @typing.override
    def inputs(self) -> cabc.Iterator[TFunctionFn]:
        finder = NestedFinder(target=TFunctionFn)
        return finder(self)


@dcls.dataclass(match_args=False)
class TFunctionFn(_TThunkBase[cabc.Callable[..., typing.Any]], TFunctionFnMixin):
    """
    `TorchFunctionT` is the thunk capturing the function calls initiated by `torch`.

    The `func` here are `torch.*` or `torch.Tensor` operators.
    """

    types: tuple[type, ...]
    "The types of the arguments."

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return _render_function_body("function", self.func, self.args, self.kwargs)


class TDispatchFnMixin(Fn, abc.ABC):
    "The mixin to define what types `Fn` graph would capture during dispatch mode."

    @typing.override
    def inputs(self) -> cabc.Iterator[TDispatchFn | FateFn]:
        finder = NestedFinder(target=TDispatchFn | FateFn)
        return finder(self)


@dcls.dataclass(match_args=False)
class TDispatchFn(_TThunkBase[_ops.OpOverload], TDispatchFnMixin):
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
        return _render_function_body("dispatch", self.func, self.args, self.kwargs)

    @property
    def is_aten(self) -> bool:
        return is_aten_op(self.func)

    @property
    def is_prim(self) -> bool:
        return is_prim_op(self.func)


def _render_function_body(
    prefix: str,
    func: cabc.Callable[..., typing.Any],
    args: tuple[typing.Any, ...],
    kwargs: dict[str, typing.Any],
) -> str:
    func_name = _render_func_name(func)
    return _render_tensor_short(prefix + "::" + func_name, args, kwargs)


def _render_func_name(func: cabc.Callable[..., typing.Any]) -> str:
    name = func.__name__

    # Only descriptors use `__get__`, and we render the descriptor itself.
    if name == "__get__":
        assert isinstance(func, types.MethodType | types.MethodWrapperType), type(func)
        return repr(func.__self__)

    # It seems that there isn't an attribute that expose the name of the `OpOverload`,
    # so here we combine `namespace` (aten, prim, ...) and `__name__` (packet.type).
    if isinstance(func, _ops.OpOverload):
        return f"torch.ops.{func.namespace}.{name}"

    # Just converting to `str` works.
    if isinstance(func, _ops.OpOverloadPacket):
        return f"torch.ops.{func!s}"

    # If it's `torch.*`.
    if getattr(torch, name, None) is func:
        return f"torch.{name}"

    # If it's `torch.Tensor.*`.
    if getattr(torch.Tensor, name, None) is func:
        return f"torch.Tensor.{name}"

    # Don't know what this is. Just use `__qualname__`.
    return func.__qualname__


@typing.no_type_check
def _render_tensor_short(
    func: str, args: tuple[typing.Any, ...], kwargs: dict[str, typing.Any]
):
    # `Attr`s are better for display than `torch.Tensor`s.

    # args = replace_tensors(args, attr)
    # kwargs = replace_tensors(kwargs, attr)

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
        thunk = TFunctionFn(func=func, types=types, args=args, kwargs=kwargs)
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

        if not all(issubclass(t, torch.Tensor) for t in types):
            raise AssertionError(f"Not all {types=} are subclasses of `torch.Tensor`.")

        thunk = TDispatchFn(func=func, args=args, kwargs=kwargs)
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


@typing.final
@dcls.dataclass(frozen=True)
class FateFn(HasParam, TDispatchFnMixin):
    """
    `FateFn` wraps a `Fate` object, which is split out so as to declutter subclasses for `Fn`.

    Each `Fate` is an implementation of an IR, and each IR can have multiple `Fate`s,
    each handling a subset of parameters (if `Fate.ok` is `False`, it's discarded.)
    """

    fate: Fate
    """
    The `Fate` object that ends up being selected.
    """

    original: TDispatchFn
    "The original `TorchDispatchFn` from which the `Fate` is translated."

    def __repr__(self) -> str:
        return repr(self.fate)

    def do(self) -> torch.Tensor:
        return self.fate.do()

    @typing.override
    def tensors(self) -> cabc.Iterator[torch.Tensor]:
        yield from find_nested_tensors(self.fate)

    @property
    def func(self):
        return self.original.func

    @property
    def args(self):
        return self.original.args

    @property
    def kwargs(self):
        return self.original.kwargs

    @classmethod
    def find_fate(cls, thunk: TDispatchFn) -> typing.Self:
        if (
            fate := find_fate(thunk.func, *thunk.args, **thunk.kwargs)
        ) is NotImplemented:
            return NotImplemented

        return cls(fate=fate, original=thunk)
