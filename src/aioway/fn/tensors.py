# Copyright (c) AIoWay Authors - All Rights Reserved

"Torch function/dispatch modes, corresponding to `__torch_function__`/`__torch_dispatch__`."

import abc
import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from torch import _ops, overrides
from torch.utils import _mode_utils as mu
from torch.utils import _python_dispatch as pyd

from aioway._common import (
    HasParam,
    find_nested_tensors,
    is_aten_op,
    is_prim_op,
    render_fcall,
    render_func_name,
    replace_tensors,
)
from aioway.fate import Fate, find_fate
from aioway.schemas import attr

from .fn import TorchThunk

__all__ = [
    "TFunctionMode",
    "TDispatchMode",
    "TFunctionFn",
    "TDispatchFn",
    "FateFn",
    "set_torch_mode",
    "torch_mode_off",
]


_do_aioway_function: bool = True
"If `False`, `TFunctionMode` won't do anything."

_do_aioway_dispatch: bool = True
"If `False`, `TDispatchMode` won't do anything."


@ctxl.contextmanager
def set_torch_mode(function: bool, dispatch: bool):
    """
    Turn on or off `__torch_function__` / `__torch_dispatch__` mode for the given scope,
    for those defined in `aioway`, depending on the flags.

    Args:
        function: Disable the `__torch_function__` mode if `True`.
        dispatch: Disable the `__torch_dispatch__` mode if `True`.

    Note:
        We are implementing this flag instead of using `no_dispatch` utility from `torch`,
        is because thier version causes segmentation fault in some cases.
    """

    global _do_aioway_function, _do_aioway_dispatch
    before = _do_aioway_function, _do_aioway_dispatch

    try:
        _do_aioway_function = function
        _do_aioway_dispatch = dispatch
        yield
    finally:
        _do_aioway_function, _do_aioway_dispatch = before


@ctxl.contextmanager
def torch_mode_off():
    with torch.DisableTorchFunction(), mu.no_dispatch():
        yield


@dcls.dataclass(match_args=False)
class TFunctionFn(TorchThunk[cabc.Callable[..., typing.Any]]):
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


@dcls.dataclass(match_args=False)
class TDispatchFn(TorchThunk[_ops.OpOverload]):
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
    func_name = render_func_name(func)
    return render_tensor_func_short(prefix + "::" + func_name, args, kwargs)


@typing.no_type_check
def replace_tensors_with_attr[T](obj: T) -> T:
    return replace_tensors(obj, attr)


def render_tensor_func_short(func: str, args, kwargs) -> str:
    # `Attr`s are better for display than `torch.Tensor`s.

    args = replace_tensors_with_attr(args)
    kwargs = replace_tensors_with_attr(kwargs)

    return render_fcall(func, *args, **kwargs)


class TFunctionMode(overrides.TorchFunctionMode, abc.ABC):
    """
    `TorchFunctionMode` is the adaptor for `torch.overrides.TorchFunctionMode`.
    """

    @abc.abstractmethod
    def __call__(self, thunk: TFunctionFn, /) -> object:
        raise NotImplementedError

    @typing.final
    @typing.override
    def __torch_function__(self, func, types, args=(), kwargs=None) -> object:
        kwargs = kwargs or {}

        if not _do_aioway_function:
            return func(*args, **kwargs)

        thunk = TFunctionFn(func=func, types=types, args=args, kwargs=kwargs)
        return self(thunk)


class TDispatchMode(pyd.TorchDispatchMode, abc.ABC):
    """
    `TorchDispatchMode` is the adaptor for `torch.data._python_dispatch.TorchDispatchMode`.
    """

    @abc.abstractmethod
    def __call__(self, thunk: TDispatchFn, /) -> object:
        raise NotImplementedError

    @typing.final
    @typing.override
    def __torch_dispatch__(self, func, types, args=(), kwargs=None) -> object:
        kwargs = kwargs or {}

        if not all(issubclass(t, torch.Tensor) for t in types):
            raise AssertionError(f"Not all {types=} are subclasses of `torch.Tensor`.")

        if not _do_aioway_dispatch:
            return func(*args, **kwargs)

        thunk = TDispatchFn(func=func, args=args, kwargs=kwargs)
        return self(thunk)


@typing.final
@dcls.dataclass(frozen=True)
class FateFn(HasParam):
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
