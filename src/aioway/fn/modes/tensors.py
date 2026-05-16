# Copyright (c) AIoWay Authors - All Rights Reserved

"Torch function/dispatch modes, corresponding to `__torch_function__`/`__torch_dispatch__`."

import abc
import contextlib as ctxl
import dataclasses as dcls
import logging
import typing
from collections import abc as cabc

import torch
from torch import _ops, overrides
from torch.utils import _python_dispatch as pyd

from aioway._common import (
    is_aten_op,
    is_prim_op,
    render_fcall,
    render_func_name,
    replace_tensors,
)
from aioway.schemas import attr

from ..fn import TorchThunk
from .toggles import OnOffCtx, OnOffStack

__all__ = [
    "TFunctionMode",
    "TDispatchMode",
    "TFunctionFn",
    "TDispatchFn",
    "set_torch_mode",
    "torch_mode_off",
    "active_dispatch_modes",
    "active_function_modes",
]

LOGGER = logging.getLogger(__name__)

FUNCTIONS: OnOffStack[TFunctionMode] = OnOffStack()
"`TFunctionMode` that is currently entered."

DISPATCHES: OnOffStack[TDispatchMode] = OnOffStack()
"`TDispatchMode` that is currently entered."


@ctxl.contextmanager
def set_torch_mode(function: bool, dispatch: bool):
    """
    Turn on or off `__torch_function__` / `__torch_dispatch__` mode for the given scope,
    for the modes that are **currently activated**.

    Args:
        function: Enable the `__torch_function__` mode if `True`.
        dispatch: Enable the `__torch_dispatch__` mode if `True`.

    Note:
        We are implementing this flag instead of using `no_dispatch` utility from `torch`,
        is because thier version causes segmentation fault in some cases.
    """

    with FUNCTIONS.switch(function), DISPATCHES.switch(dispatch):
        yield


@ctxl.contextmanager
def torch_mode_off():
    with set_torch_mode(False, False):
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


type _Mode = overrides.TorchFunctionMode | pyd.TorchDispatchMode


@dcls.dataclass
class TModeOnOff[T](OnOffCtx, abc.ABC):
    """
    The mixin for either `TFunctionMode`, `TDispatchMode`.
    """

    _TORCH_MODE: typing.ClassVar[cabc.Callable[..., _Mode]]
    """
    The actual context passed to `torch`.
    These are specific modes that honor the `on` switch (hence private function).
    """

    @abc.abstractmethod
    def __call__(self, thunk: T, /) -> object:
        raise NotImplementedError

    @typing.override
    @ctxl.contextmanager
    def ctx(self: typing.Self):
        """
        Enter the `__torch_function__` / `__torch_dispatch__` context,
        and store the mode itself s.t. it can be turned on / off later.
        """

        with self.STACK.hold(self), self._TORCH_MODE(self):
            yield self


@typing.final
class _TFunctionModeCtx(overrides.TorchFunctionMode):
    "The `__torch_function__` adaptor"

    def __init__(self, mode: TFunctionMode) -> None:
        super().__init__()
        self.mode = mode

    @typing.final
    @typing.override
    def __torch_function__(self, func, types, args=(), kwargs=None) -> object:
        kwargs = kwargs or {}

        # The mode can be turned off.
        if not self.mode.on:
            return func(*args, **kwargs)

        thunk = TFunctionFn(func=func, types=types, args=args, kwargs=kwargs)
        return self.mode(thunk)


@dcls.dataclass
class TFunctionMode(TModeOnOff[TFunctionFn], abc.ABC):
    """
    `TFunctionMode` is the adaptor for `torch.overrides.TorchFunctionMode`.

    It provides a `ctx` context manager that is responsible for
    entering and exiting the torch mode context, as well as an `on` switch.
    """

    STACK: typing.ClassVar = FUNCTIONS
    _TORCH_MODE: typing.ClassVar = _TFunctionModeCtx


@typing.final
class _TDispatchModeCtx(pyd.TorchDispatchMode):
    "The `__torch_dispatch__` adaptor"

    def __init__(self, mode: TDispatchMode) -> None:
        super().__init__()
        self.mode = mode

    @typing.final
    @typing.override
    def __torch_dispatch__(self, func, types, args=(), kwargs=None) -> object:
        kwargs = kwargs or {}

        if not all(issubclass(t, torch.Tensor) for t in types):
            raise AssertionError(f"Not all {types=} are subclasses of `torch.Tensor`.")

        # The mode can be turned off.
        if not self.mode.on:
            return func(*args, **kwargs)

        thunk = TDispatchFn(func=func, args=args, kwargs=kwargs)
        return self.mode(thunk)


@dcls.dataclass
class TDispatchMode(TModeOnOff[TDispatchFn], abc.ABC):
    """
    `TDispatchMode` is the adaptor for `torch.data._python_dispatch.TorchDispatchMode`.

    It provides a `ctx` context manager that is responsible for
    entering and exiting the torch mode context, as well as an `on` switch.
    """

    STACK: typing.ClassVar = DISPATCHES
    _TORCH_MODE: typing.ClassVar = _TDispatchModeCtx


def active_function_modes():
    return FUNCTIONS


def active_dispatch_modes():
    return DISPATCHES
