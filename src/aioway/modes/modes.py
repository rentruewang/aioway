# Copyright (c) AIoWay Authors - All Rights Reserved

"The base classes for modes."

import abc
import contextlib as ctxl
import dataclasses as dcls
import functools
import logging
import typing
import warnings
from collections import abc as cabc

from torch import _ops, overrides
from torch.utils import _python_dispatch as pyd

from aioway._utils import Stack, find_nested_tensors
from aioway.schemas import render_function_body_prefix

from .aten import is_aten_op, is_prim_op

__all__ = [
    "Mode",
    "ModeCtx",
    "ModeStack",
    "ModeThunk",
    "TorchFuncMode",
    "TorchDispMode",
    "TorchFuncThunk",
    "TorchDispThunk",
    "functions",
    "dispatches",
]

LOGGER = logging.getLogger(__name__)


@functools.cache
def functions() -> ModeStack[TorchFuncMode]:
    "`TorchFuncMode` that is currently entered."
    return ModeStack()


@functools.cache
def dispatches() -> ModeStack[TorchDispMode]:
    "`TorchDispMode` that is currently entered."
    return ModeStack()


class ModeThunk[**P = ..., T = typing.Any]:
    """
    `ModeThunk` is a basic `Thunk` used by the modes to store a function call,
    s.t. overriding can have more elegant function signature.
    """

    def __init__(
        self, func: cabc.Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> None:
        if not callable(func):
            raise TypeError(f"{func=} is not callable.")

        self._func = func
        self._args = args
        self._kwargs = kwargs

    def __call__(self):
        return self.func(*self.args, **self.kwargs)

    def inputs(self):
        yield from find_nested_tensors(self.args)
        yield from find_nested_tensors(self.kwargs)

    @property
    def func(self) -> cabc.Callable[P, T]:
        "The function to call. Must be callable."
        return self._func

    @property
    def args(self) -> typing.Any:
        "The positional args."
        return self._args

    @property
    def kwargs(self) -> typing.Any:
        "The keyword arguments."
        return self._kwargs

    @property
    def requires_grad(self) -> bool:
        "Check if any of the inputs requires grad."

        return any(tensor.requires_grad for tensor in self.inputs())


class ModeCtx[T: ModeThunk](abc.ABC):
    "The adaptor for torch contexts (acts like a torch mode to the type checker)."

    if typing.TYPE_CHECKING:

        def __enter__(self) -> typing.Any: ...

        def __exit__(self, *args, **kwargs) -> typing.Any: ...

    def __init__(self, mode: Mode[T]) -> None:
        super().__init__()

        self.mode = mode

        if not isinstance(self, ctxl.AbstractContextManager):
            raise ValueError(f"{type(self)=} is not a context manager.")

    def __torch_call__(self, name: str, func, types, args=(), kwargs=None) -> object:
        kwargs = kwargs or {}

        # Since we mimick torch's execution model (our stack is synced with theirs),
        # where a mode is pushed onto a stack during `with` and borrowed whenever invoked,
        # torch only will invoke those modes when they are the top most mode on the stack.
        #
        # Here we check if `__torch_*__` call is synced to ours (that only top is called).
        # If fail, our assumption about prioritizing the top mode on stack is incorrect.
        # A warning is emitted to notify us that we need to update the execution model.
        if self.mode.STACK.top() is not self.mode:
            warnings.warn(
                f"Modes mismatch in `{name}`. "
                f"`torch` has changed their `{name}` execution model. "
                "This may cause bugs.",
                RuntimeWarning,
            )

        with self.mode.STACK.borrow() as mode:
            assert self.mode is mode

            # The mode can be turned off.
            if not self.mode.on:
                return func(*args, **kwargs)

            return self._impl(func, types, args, kwargs)

    @abc.abstractmethod
    def _impl(self, func, types, args, kwargs) -> object:
        raise NotImplementedError


@dcls.dataclass
class Mode[T: ModeThunk = ModeThunk, V = object](abc.ABC):
    """
    `Mode` is a mixin class that gives the subclasses a toggle.

    Calls `.activate()` to enter the scope.
    """

    STACK: typing.ClassVar[ModeStack]
    "The stack."

    _TORCH_MODE: typing.ClassVar[cabc.Callable[..., ModeCtx]]
    """
    The actual context passed to `torch`.
    These are specific modes that honor the `on` switch (hence private function).
    """

    _: dcls.KW_ONLY

    on: bool = True
    "The toggle to control whether or not to run the current mode."

    @ctxl.contextmanager
    def activate(self) -> cabc.Generator[typing.Self]:
        """
        The context manager that can be entered, and will be constrained by `self.on`.

        I'm using this function as public API because I don't like `__enter__`, `__exit__`,
        which is much less elegant than `ctxl.contextmanager` (I know it's necessary).
        """

        with self.STACK.hold(self), self._torch_mode():
            yield self

    @abc.abstractmethod
    def run(self, thunk: T, /) -> V:
        """
        The overriding function that customizes `thunk()`.
        Calling `thunk()` should run the next `Mode.run` until the `STACK` is exhausted,
        """

        raise NotImplementedError

    @ctxl.contextmanager
    def switch(self, on: bool, /):
        "Switch to `on` in the scope (can be overwritten)."

        before = self.on
        self.on = on
        try:
            yield
        finally:
            self.on = before

    @abc.abstractmethod
    def _torch_mode(self) -> ModeCtx:
        """
        The actual context passed to `torch`.
        These are specific modes that honor the `on` switch (hence private function).
        """

        raise NotImplementedError


class ModeStack[T: Mode[typing.Any, typing.Any]](Stack[T]):
    """
    `OnOffStack` provides additional utilites to decide when to turn on or off.
    """

    @ctxl.contextmanager
    def switch(self, on: bool | list[bool], /):
        """
        Temporarily set the `on` switch to the value given.
        """

        before = self.on
        self.on = on

        try:
            yield
        finally:
            self.on = before

    @property
    def on(self) -> list[bool]:
        "Get the on off values."

        return [frame.on for frame in self]

    @on.setter
    def on(self, to: bool | list[bool]) -> None:
        LOGGER.debug("Current stack %s", self)
        LOGGER.debug("Setting to %s", to)

        if isinstance(to, bool):
            to = [to] * len(self)

        if len(to) != len(self):
            raise ValueError(f"Value {to=} should have equal length with {self=}.")

        for frame, val in zip(self, to):
            frame.on = val

        LOGGER.debug("Status after setting %s", self)


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

        assert self.mode.STACK is functions()

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

        assert self.mode.STACK is dispatches()

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

    STACK: typing.ClassVar = functions()

    @typing.override
    def _torch_mode(self) -> _TorchFuncModeCtx:
        return _TorchFuncModeCtx(self)


@dcls.dataclass
class TorchDispMode(Mode[TorchDispThunk], abc.ABC):
    """
    `TorchDispMode` is the adaptor for `torch.data._python_dispatch.TorchDispatchMode`.

    It provides a `ctx` context manager that is responsible for
    entering and exiting the torch mode context, as well as an `on` switch.
    """

    STACK: typing.ClassVar = dispatches()

    @typing.override
    def _torch_mode(self) -> _TorchDispModeCtx:
        return _TorchDispModeCtx(self)
