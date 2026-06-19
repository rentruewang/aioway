# Copyright (c) AIoWay Authors - All Rights Reserved

"Torch function/dispatch modes, corresponding to `__torch_function__`/`__torch_dispatch__`."

import abc
import contextlib as ctxl
import dataclasses as dcls
import logging
import typing
import warnings
from collections import abc as cabc

import torch
from torch import _ops, overrides
from torch.utils import _python_dispatch as pyd

from aioway._comps import TorchThunk
from aioway._utils import is_aten_op, is_prim_op, render_function_body_prefix

from .modes import Mode, ModeStack

__all__ = ["TorchFuncMode", "TorchDispMode", "TorchFuncThunk", "TorchDispThunk"]

LOGGER = logging.getLogger(__name__)

FUNCTIONS: ModeStack[TorchFuncMode] = ModeStack()
"`TorchFuncMode` that is currently entered."

DISPATCHES: ModeStack[TorchDispMode] = ModeStack()
"`TorchDispMode` that is currently entered."


@typing.final
class TorchFuncThunk[**P = ...](TorchThunk):
    """
    `TorchFuncThunk` is the thunk capturing the function calls initiated by `torch`.

    The `func` here are `torch.*` or `torch.Tensor` operators.
    """

    def __init__(
        self,
        func: cabc.Callable[P, object],
        types: tuple[type, ...],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        super().__init__(func, *args, **kwargs)

        self._types = types

        if not (
            True
            and isinstance(self.types, tuple)
            and all(isinstance(t, type) for t in self.types)
        ):
            raise TypeError(f"{self.types=} should be a tuple of types.")

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return render_function_body_prefix(
            "function", self.func, self.args, self.kwargs
        )

    @property
    def types(self) -> tuple[type, ...]:
        "The types of the arguments."
        return self._types


@typing.final
class TorchDispThunk(TorchThunk):
    """
    `TorchDispThunk` is the thunk capturing the function calls initiated by `torch`.
    This is by default what a null-op `__torch_dispatch__` would call.

    The `func` here are `torch.ops.aten.*` operators.
    """

    if typing.TYPE_CHECKING:

        @property
        def func(self) -> _ops.OpOverload: ...

    def __init__(self, func: _ops.OpOverload, *args, **kwargs) -> None:
        super().__init__(func, *args, **kwargs)

        if not isinstance(func, _ops.OpOverload):
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
class TorchModeOnOff[T: TorchThunk](Mode[T, object], abc.ABC):
    """
    The mixin for either `TorchFuncMode`, `TorchDispMode`.
    """

    _TORCH_MODE: typing.ClassVar[cabc.Callable[..., _Mode]]
    """
    The actual context passed to `torch`.
    These are specific modes that honor the `on` switch (hence private function).
    """

    @typing.override
    @ctxl.contextmanager
    def enter(self) -> cabc.Generator[None]:
        """
        Enter the `__torch_function__` / `__torch_dispatch__` context,
        and store the mode itself s.t. it can be turned on / off later.
        """

        with self._TORCH_MODE(self):
            yield


@typing.final
class _TorchFuncModeCtx(overrides.TorchFunctionMode):
    "The `__torch_function__` adaptor."

    def __init__(self, mode: TorchFuncMode) -> None:
        super().__init__()
        self.mode = mode

        assert self.mode.STACK is FUNCTIONS

    @typing.final
    @typing.override
    def __torch_function__(self, func, types, args=(), kwargs=None) -> object:
        kwargs = kwargs or {}

        with FUNCTIONS.borrow() as mode:
            if mode is not self.mode:
                warnings.warn(
                    "Modes mismatch in `__torch_function__`. "
                    "`torch` has changed their `__torch_function__` execution model. "
                    "This may cause bugs.",
                    RuntimeWarning,
                )

            return self.__impl(func, types, args, kwargs)

    def __impl(self, func, types, args, kwargs) -> object:

        # The mode can be turned off.
        if not self.mode.on:
            return func(*args, **kwargs)

        thunk = TorchFuncThunk(func, types, *args, **kwargs)
        return self.mode.run(thunk)


@dcls.dataclass
class TorchFuncMode(TorchModeOnOff[TorchFuncThunk], abc.ABC):
    """
    `TorchFuncMode` is the adaptor for `torch.overrides.TorchFunctionMode`.

    It provides a `ctx` context manager that is responsible for
    entering and exiting the torch mode context, as well as an `on` switch.
    """

    STACK: typing.ClassVar = FUNCTIONS
    _TORCH_MODE: typing.ClassVar = _TorchFuncModeCtx


@typing.final
class _TorchDispModeCtx(pyd.TorchDispatchMode):
    "The `__torch_dispatch__` adaptor."

    def __init__(self, mode: TorchDispMode) -> None:
        super().__init__()
        self.mode = mode

        assert self.mode.STACK is DISPATCHES

    @typing.final
    @typing.override
    def __torch_dispatch__(self, func, types, args=(), kwargs=None) -> object:
        kwargs = kwargs or {}

        if not all(issubclass(t, torch.Tensor) for t in types):
            raise AssertionError(f"Not all {types=} are subclasses of `torch.Tensor`.")

        with DISPATCHES.borrow() as mode:
            if mode is not self.mode:
                warnings.warn(
                    "Modes mismatch in `__torch_dispatch__`. "
                    "`torch` has changed their `__torch_dispatch__` execution model. "
                    "This may cause bugs.",
                    RuntimeWarning,
                )

            return self.__impl(func, args, kwargs)

    def __impl(self, func, args, kwargs) -> object:
        # The mode can be turned off.
        if not self.mode.on:
            return func(*args, **kwargs)

        thunk = TorchDispThunk(func, *args, **kwargs)
        return self.mode.run(thunk)


@dcls.dataclass
class TorchDispMode(TorchModeOnOff[TorchDispThunk], abc.ABC):
    """
    `TorchDispMode` is the adaptor for `torch.data._python_dispatch.TorchDispatchMode`.

    It provides a `ctx` context manager that is responsible for
    entering and exiting the torch mode context, as well as an `on` switch.
    """

    STACK: typing.ClassVar = DISPATCHES
    _TORCH_MODE: typing.ClassVar = _TorchDispModeCtx
