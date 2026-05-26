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

from aioway._fn import TorchThunk, thunk_dcls
from aioway._torch import is_aten_op, is_prim_op

from ._on_off import OnOffCtx, OnOffStack
from .common import render_function_body_prefix

__all__ = ["TorchFuncMode", "TorchDispMode", "TorchFuncFn", "TorchDispFn"]

LOGGER = logging.getLogger(__name__)

FUNCTIONS: OnOffStack[TorchFuncMode] = OnOffStack()
"`TorchFuncMode` that is currently entered."

DISPATCHES: OnOffStack[TorchDispMode] = OnOffStack()
"`TorchDispMode` that is currently entered."


@typing.final
@thunk_dcls
class TorchFuncFn(TorchThunk[cabc.Callable[..., typing.Any]]):
    """
    `TorchFuncFn` is the thunk capturing the function calls initiated by `torch`.

    The `func` here are `torch.*` or `torch.Tensor` operators.
    """

    types: tuple[type, ...]
    "The types of the arguments."

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return render_function_body_prefix(
            "function", self.func, self.args, self.kwargs
        )


@typing.final
@thunk_dcls
class TorchDispFn(TorchThunk[_ops.OpOverload]):
    """
    `TorchDispFn` is the thunk capturing the function calls initiated by `torch`.
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
        return render_function_body_prefix(
            "dispatch", self.func, self.args, self.kwargs
        )

    @property
    def is_aten(self) -> bool:
        return is_aten_op(self.func)

    @property
    def is_prim(self) -> bool:
        return is_prim_op(self.func)


type _Mode = overrides.TorchFunctionMode | pyd.TorchDispatchMode


@dcls.dataclass
class TorchModeOnOff[T](OnOffCtx, abc.ABC):
    """
    The mixin for either `TorchFuncMode`, `TorchDispMode`.
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
    def enter(self: typing.Self):
        """
        Enter the `__torch_function__` / `__torch_dispatch__` context,
        and store the mode itself s.t. it can be turned on / off later.
        """

        with self.STACK.hold(self), self._TORCH_MODE(self):
            yield self


@typing.final
class _TorchFuncModeCtx(overrides.TorchFunctionMode):
    "The `__torch_function__` adaptor"

    def __init__(self, mode: TorchFuncMode) -> None:
        super().__init__()
        self.mode = mode

    @typing.final
    @typing.override
    def __torch_function__(self, func, types, args=(), kwargs=None) -> object:
        kwargs = kwargs or {}

        # The mode can be turned off.
        if not self.mode.on:
            return func(*args, **kwargs)

        thunk = TorchFuncFn(func=func, types=types, args=args, kwargs=kwargs)
        return self.mode(thunk)


@dcls.dataclass
class TorchFuncMode(TorchModeOnOff[TorchFuncFn], abc.ABC):
    """
    `TorchFuncMode` is the adaptor for `torch.overrides.TorchFunctionMode`.

    It provides a `ctx` context manager that is responsible for
    entering and exiting the torch mode context, as well as an `on` switch.
    """

    STACK: typing.ClassVar = FUNCTIONS
    _TORCH_MODE: typing.ClassVar = _TorchFuncModeCtx


@typing.final
class _TorchDispModeCtx(pyd.TorchDispatchMode):
    "The `__torch_dispatch__` adaptor"

    def __init__(self, mode: TorchDispMode) -> None:
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

        thunk = TorchDispFn(func=func, args=args, kwargs=kwargs)
        return self.mode(thunk)


@dcls.dataclass
class TorchDispMode(TorchModeOnOff[TorchDispFn], abc.ABC):
    """
    `TorchDispMode` is the adaptor for `torch.data._python_dispatch.TorchDispatchMode`.

    It provides a `ctx` context manager that is responsible for
    entering and exiting the torch mode context, as well as an `on` switch.
    """

    STACK: typing.ClassVar = DISPATCHES
    _TORCH_MODE: typing.ClassVar = _TorchDispModeCtx
