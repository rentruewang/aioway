# Copyright (c) AIoWay Authors - All Rights Reserved

"Some pre-bundled implementations for thunks."

import dataclasses as dcls
import typing

import loguru as L
import rich

from aioway.torch._utils import replace_tensors

from .aten import AtenThunk
from .attrs import replace_tensors_with_attr
from .overrides import (
    TorchDispMode,
    TorchDispThunk,
    TorchFuncMode,
    TorchFuncThunk,
    is_fake_mode_on,
)

__all__ = [
    "route_aten_thunk",
    "clone_in_fake_mode",
    "PrintTorchFunc",
    "PrintTorchDisp",
    "LogTorchFunc",
    "LogTorchDisp",
]


@TorchDispMode.function
def route_aten_thunk(thunk: TorchDispThunk) -> object:
    """
    Route `torch.aten` calls to `AtenThunk` for some `aioway` specific functionalities.
    """

    fn: AtenThunk | TorchDispThunk

    if (found := AtenThunk.from_thunk(thunk)) is not None:
        fn = found

    # Cannot find corresponding operator, set it to the input `thunk`.
    else:
        fn = thunk

    assert isinstance(fn, TorchDispThunk | AtenThunk), type(fn)

    # Here, `AtenThunk` would do its magic and overwrite functions.
    try:
        return fn()
    except Exception as e:
        L.logger.error("{} raises an error.", fn)
        raise RuntimeError(f"{fn!r}") from e


@TorchDispMode.function
def clone_in_fake_mode(thunk: TorchDispThunk) -> object:
    """
    Automatically call `.clone()` on all tensors in the torch dispatch mode.

    This is useful to force a new `id` s.t. the tracking won't fail.
    """
    result = thunk()

    if not is_fake_mode_on():
        return result

    # In fake mode, clone the tensor to prevent `FakeTensor` reuse. Should be cheap.
    return replace_tensors(result, lambda tensor: tensor.clone())


class _HasRichFlagMixin:
    def __init__(self, rich: bool = False) -> None:
        super().__init__()
        self._rich = rich


class PrintTorchFunc(_HasRichFlagMixin, TorchFuncMode):
    @typing.override
    def run(self, thunk: TorchFuncThunk, /) -> object:
        return _TorchThunkPrinter(rich=self._rich)(thunk)


class PrintTorchDisp(_HasRichFlagMixin, TorchDispMode):
    @typing.override
    def run(self, thunk: TorchDispThunk, /) -> object:
        return _TorchThunkPrinter(rich=self._rich)(thunk)


@dcls.dataclass(frozen=True)
class _TorchThunkPrinter:
    rich: bool
    "Use rich for printing."

    def __call__(self, thunk: TorchFuncThunk | TorchDispThunk) -> object:
        self.print("invoke", thunk)
        result = thunk()
        self.print("return", thunk, "->", replace_tensors_with_attr(result))
        return result

    @property
    def print(self):
        return rich.print if self.rich else print


@dcls.dataclass
class LogTorchFunc(TorchFuncMode):
    """
    Log every call to function mode.
    """

    level: int
    "The level to log to."

    @typing.override
    def run(self, thunk: TorchFuncThunk) -> object:
        result = thunk()
        L.logger.log(self.level, "{}", thunk)
        return result


@dcls.dataclass
class LogTorchDisp(TorchDispMode):
    """
    Log every call to dispatch mode.
    """

    level: int
    "The level to log to."

    @typing.override
    def run(self, thunk: TorchDispThunk) -> object:
        result = thunk()
        L.logger.log(self.level, "{}", thunk)
        return result
