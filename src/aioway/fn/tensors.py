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
    HasParam,
    Stack,
    find_nested_tensors,
    is_aten_op,
    is_prim_op,
    render_fcall,
    render_func_name,
    replace_tensors,
)
from aioway.fate import Fate, find_fate
from aioway.schemas import attr

from .fn import Fn, TorchThunk

__all__ = [
    "TFunctionMode",
    "TDispatchMode",
    "TFunctionFn",
    "TDispatchFn",
    "FateFn",
    "set_torch_mode",
    "torch_mode_off",
    "active_dispatch_modes",
    "active_function_modes",
]

LOGGER = logging.getLogger(__name__)

_FUNCTIONS: Stack[TFunctionMode] = Stack()
"`TFunctionFn` that is currently in scope."

_DISPATCHES: Stack[TDispatchMode] = Stack()
"`TDispatchFn` that is currently in scope."


@ctxl.contextmanager
def set_torch_mode(function: bool, dispatch: bool):
    """
    Turn on or off `__torch_function__` / `__torch_dispatch__` mode for the given scope,
    for the modes that are **currently activated**.

    Args:
        function: Disable the `__torch_function__` mode if `True`.
        dispatch: Disable the `__torch_dispatch__` mode if `True`.

    Note:
        We are implementing this flag instead of using `no_dispatch` utility from `torch`,
        is because thier version causes segmentation fault in some cases.
    """

    functions_before = _get_stack_on_off(_FUNCTIONS)
    dispatches_before = _get_stack_on_off(_DISPATCHES)

    _set_stack_on_off(_FUNCTIONS, function)
    _set_stack_on_off(_DISPATCHES, dispatch)

    try:
        yield
    finally:
        _set_stack_on_off(_FUNCTIONS, functions_before)
        _set_stack_on_off(_DISPATCHES, dispatches_before)


@ctxl.contextmanager
def torch_mode_off():
    with set_torch_mode(False, False):
        yield


def _get_stack_on_off[M: TorchModeContextMixin](stack: Stack[M]):
    return [frame.on for frame in stack]


def _set_stack_on_off[M: TorchModeContextMixin](stack: Stack[M], to: bool | list[bool]):
    LOGGER.debug("Current stack %s", stack)
    LOGGER.debug("Status before setting %s", _get_stack_on_off(stack))
    LOGGER.debug("Setting to %s", to)

    if isinstance(to, bool):
        to = [to] * len(stack)

    if len(to) != len(stack):
        raise ValueError(f"Value {to=} should have equal length with {stack=}.")

    for frame, val in zip(stack, to):
        frame.on = val

    LOGGER.debug("Status after setting %s", _get_stack_on_off(stack))


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
class TorchModeContextMixin:
    """
    The mixin for either `TFunctionMode`, `TDispatchMode`.
    """

    STACK: typing.ClassVar[Stack[typing.Self]]
    "The stack. One of `_FUNCTIONS`, `_DISPATCHES`."

    _TORCH_MODE: typing.ClassVar[cabc.Callable[..., _Mode]]
    """
    The actual context passed to `torch`.
    These are specific modes that honor the `on` switch (hence private function).
    """

    _: dcls.KW_ONLY

    on: bool = True
    "The toggle to control whether or not to run the current mode."

    @ctxl.contextmanager
    def ctx(self: typing.Self):
        """
        Enter the `__torch_function__` / `__torch_dispatch__` context,
        and store the mode itself s.t. it can be turned on / off later.
        """

        self.STACK.append(self)
        try:
            with self._TORCH_MODE(self):
                yield self
        finally:
            _ = self.STACK.pop()

    @ctxl.contextmanager
    def switch(self, on: bool):
        "Switch to `on` in the scope (can be overwritten)."

        before = self.on
        self.on = on
        try:
            yield
        finally:
            self.on = before


@typing.final
class _TorchFunctionModeCtx(overrides.TorchFunctionMode):
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
class TFunctionMode(TorchModeContextMixin, abc.ABC):
    """
    `TFunctionMode` is the adaptor for `torch.overrides.TorchFunctionMode`.

    It provides a `context` context manager that is responsible for
    entering and exiting the torch mode context, as well as an `on` switch.
    """

    STACK: typing.ClassVar = _FUNCTIONS
    _TORCH_MODE: typing.ClassVar = _TorchFunctionModeCtx

    @abc.abstractmethod
    def __call__(self, thunk: TFunctionFn, /) -> object:
        raise NotImplementedError


@typing.final
class _TorchDispatchModeCtx(pyd.TorchDispatchMode):
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
class TDispatchMode(TorchModeContextMixin, abc.ABC):
    """
    `TorchDispatchMode` is the adaptor for `torch.data._python_dispatch.TorchDispatchMode`.
    """

    STACK: typing.ClassVar = _DISPATCHES
    _TORCH_MODE: typing.ClassVar = _TorchDispatchModeCtx

    @abc.abstractmethod
    def __call__(self, thunk: TDispatchFn, /) -> object:
        raise NotImplementedError


@typing.final
@dcls.dataclass(frozen=True)
class FateFn(HasParam, Fn):
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

    @typing.override
    def do(self) -> object:
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
        fate = find_fate(thunk.func, *thunk.args, **thunk.kwargs)

        if fate is NotImplemented:
            return NotImplemented

        else:
            return cls(fate=fate, original=thunk)


def active_function_modes():
    return _FUNCTIONS


def active_dispatch_modes():
    return _DISPATCHES
