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

from aioway._common import is_aten_op, is_prim_op

from ..common import render_function_body_prefix
from ..fn import TorchThunk
from ._on_off import OnOffCtx, OnOffStack

__all__ = [
    "TorFuncMode",
    "TorDisMode",
    "TorFuncFn",
    "TorDisFn",
    "set_torch_mode",
    "torch_mode_off",
]

LOGGER = logging.getLogger(__name__)

FUNCTIONS: OnOffStack[TorFuncMode] = OnOffStack()
"`TorFuncMode` that is currently entered."

DISPATCHES: OnOffStack[TorDisMode] = OnOffStack()
"`TorDisMode` that is currently entered."


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
class TorFuncFn(TorchThunk[cabc.Callable[..., typing.Any]]):
    """
    `TorFuncFn` is the thunk capturing the function calls initiated by `torch`.

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


@dcls.dataclass(match_args=False)
class TorDisFn(TorchThunk[_ops.OpOverload]):
    """
    `TorDisFn` is the thunk capturing the function calls initiated by `torch`.
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
class TModeOnOff[T](OnOffCtx, abc.ABC):
    """
    The mixin for either `TorFuncMode`, `TorDisMode`.
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
class _TorFuncModeCtx(overrides.TorchFunctionMode):
    "The `__torch_function__` adaptor"

    def __init__(self, mode: TorFuncMode) -> None:
        super().__init__()
        self.mode = mode

    @typing.final
    @typing.override
    def __torch_function__(self, func, types, args=(), kwargs=None) -> object:
        kwargs = kwargs or {}

        # The mode can be turned off.
        if not self.mode.on:
            return func(*args, **kwargs)

        thunk = TorFuncFn(func=func, types=types, args=args, kwargs=kwargs)
        return self.mode(thunk)


@dcls.dataclass
class TorFuncMode(TModeOnOff[TorFuncFn], abc.ABC):
    """
    `TorFuncMode` is the adaptor for `torch.overrides.TorchFunctionMode`.

    It provides a `ctx` context manager that is responsible for
    entering and exiting the torch mode context, as well as an `on` switch.
    """

    STACK: typing.ClassVar = FUNCTIONS
    _TORCH_MODE: typing.ClassVar = _TorFuncModeCtx


@typing.final
class _TorDisModeCtx(pyd.TorchDispatchMode):
    "The `__torch_dispatch__` adaptor"

    def __init__(self, mode: TorDisMode) -> None:
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

        thunk = TorDisFn(func=func, args=args, kwargs=kwargs)
        return self.mode(thunk)


@dcls.dataclass
class TorDisMode(TModeOnOff[TorDisFn], abc.ABC):
    """
    `TorDisMode` is the adaptor for `torch.data._python_dispatch.TorchDispatchMode`.

    It provides a `ctx` context manager that is responsible for
    entering and exiting the torch mode context, as well as an `on` switch.
    """

    STACK: typing.ClassVar = DISPATCHES
    _TORCH_MODE: typing.ClassVar = _TorDisModeCtx
