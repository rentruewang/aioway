# Copyright (c) AIoWay Authors - All Rights Reserved

"Torch function/dispatch modes, corresponding to `__torch_function__`/`__torch_dispatch__`."

import abc
import dataclasses as dcls
import logging
import typing
from collections import abc as cabc

from torch import _ops, overrides
from torch.utils import _python_dispatch as pyd

from aioway._utils import is_aten_op, is_prim_op, render_function_body_prefix

from .modes import Mode, ModeCtx, ModeStack, ModeThunk

__all__ = ["TorchFuncMode", "TorchDispMode", "TorchFuncThunk", "TorchDispThunk"]

LOGGER = logging.getLogger(__name__)

FUNCTIONS: ModeStack[TorchFuncMode] = ModeStack()
"`TorchFuncMode` that is currently entered."

DISPATCHES: ModeStack[TorchDispMode] = ModeStack()
"`TorchDispMode` that is currently entered."


@typing.final
class TorchFuncThunk[**P = ...](ModeThunk):
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
class TorchDispThunk(ModeThunk):
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


@typing.final
class _TorchFuncModeCtx(ModeCtx[TorchFuncThunk], overrides.TorchFunctionMode):
    "The `__torch_function__` adaptor."

    def __init__(self, mode: TorchFuncMode) -> None:
        super().__init__(mode)

        assert self.mode.STACK is FUNCTIONS

    @typing.final
    @typing.override
    def __torch_function__(self, func, types, args=(), kwargs=None) -> object:
        return self.__torch_call__("__torch_function__", func, types, args, kwargs)

    @typing.override
    def _impl(self, func, types, args, kwargs) -> object:
        thunk = TorchFuncThunk(func, types, *args, **kwargs)
        return self.mode.run(thunk)


@typing.final
class _TorchDispModeCtx(ModeCtx[TorchDispThunk], pyd.TorchDispatchMode):
    "The `__torch_dispatch__` adaptor."

    def __init__(self, mode: TorchDispMode) -> None:
        super().__init__(mode)

        assert self.mode.STACK is DISPATCHES

    @typing.final
    @typing.override
    def __torch_dispatch__(self, func, types, args=(), kwargs=None) -> object:
        return self.__torch_call__("__torch_dispatch__", func, types, args, kwargs)

    @typing.override
    def _impl(self, func, types, args, kwargs) -> object:
        _ = types
        thunk = TorchDispThunk(func, *args, **kwargs)
        return self.mode.run(thunk)


@dcls.dataclass
class TorchFuncMode(Mode[TorchFuncThunk], abc.ABC):
    """
    `TorchFuncMode` is the adaptor for `torch.overrides.TorchFunctionMode`.

    It provides a `ctx` context manager that is responsible for
    entering and exiting the torch mode context, as well as an `on` switch.
    """

    STACK: typing.ClassVar = FUNCTIONS
    _TORCH_MODE: typing.ClassVar = _TorchFuncModeCtx


@dcls.dataclass
class TorchDispMode(Mode[TorchDispThunk], abc.ABC):
    """
    `TorchDispMode` is the adaptor for `torch.data._python_dispatch.TorchDispatchMode`.

    It provides a `ctx` context manager that is responsible for
    entering and exiting the torch mode context, as well as an `on` switch.
    """

    STACK: typing.ClassVar = DISPATCHES
    _TORCH_MODE: typing.ClassVar = _TorchDispModeCtx
