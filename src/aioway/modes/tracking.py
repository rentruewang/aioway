# Copyright (c) AIoWay Authors - All Rights Reserved

"Tracking / routing related `Thunk`s."

from pydantic_settings.sources.providers.toml import import_toml
import contextlib as ctxl
import dataclasses as dcls
import logging
import typing

import rich

from aioway.tensors import (
    AtenThunk,
    TorchDispMode,
    TorchDispThunk,
    TorchFuncMode,
    TorchFuncThunk,
    fake_mode,
    is_fake_mode_on,
    replace_tensors,
    replace_tensors_with_attr,
)

from .hists import HistTensorGraph

__all__ = [
    "track_torch_thunks",
    "track_torch_fake_thunks",
    "PrintTorchFunc",
    "PrintTorchDisp",
    "LogTorchFunc",
    "LogTorchDis",
    "TrackTorchDispHist",
    "TrackTorchFuncHist",
    "clone_in_fake_mode",
    "route_aten_thunk",
]

LOGGER = logging.getLogger(__name__)


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

    logger: logging.Logger = LOGGER
    "The logger to log to. Default to the one in the current module."

    @typing.override
    def run(self, thunk: TorchFuncThunk) -> object:
        result = thunk()
        self.logger.log(self.level, "%s", thunk)
        return result


@dcls.dataclass
class LogTorchDis(TorchDispMode):
    """
    Log every call to dispatch mode.
    """

    level: int
    "The level to log to."

    logger: logging.Logger = LOGGER
    "The logger to log to. Default to the one in the current module."

    @typing.override
    def run(self, thunk: TorchDispThunk) -> object:
        result = thunk()
        self.logger.log(self.level, "%s", thunk)
        return result


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
        LOGGER.error("%r raises an error.", fn)
        raise RuntimeError(f"{fn!r}") from e


@dcls.dataclass
class TrackTorchDispHist(TorchDispMode):
    "Track torch dispatch history."

    history: HistTensorGraph[TorchDispThunk | AtenThunk] = dcls.field(
        default_factory=HistTensorGraph
    )
    "The history used for tracking."

    def run(self, thunk: TorchDispThunk) -> object:
        return self.history.execute(thunk)


@dcls.dataclass
class TrackTorchFuncHist(TorchFuncMode):
    """
    Saves the intermediate graph into a `FnHistory` object.
    """

    history: HistTensorGraph[TorchFuncThunk] = dcls.field(
        default_factory=HistTensorGraph
    )
    """
    The `HistTensorGraph` instance that would be responsible for tracking history,
    and which provides a graph API to interact with saved tensors.
    """

    @typing.override
    def run(self, thunk: TorchFuncThunk, /) -> object:
        return self.history.execute(thunk)


class HistoryCollection(typing.NamedTuple):
    function: HistTensorGraph[TorchFuncThunk]
    dispatch: HistTensorGraph[TorchDispThunk | AtenThunk]


@ctxl.contextmanager
def track_torch_thunks():
    """
    Track all calls into the torch dispatch mode.
    """

    dis = TrackTorchDispHist()
    func = TrackTorchFuncHist()

    with func.activate(), dis.activate(), route_aten_thunk.activate():
        yield HistoryCollection(function=func.history, dispatch=dis.history)


@ctxl.contextmanager
def track_torch_fake_thunks():
    """
    Track all calls into the torch dispatch mode, and activate fake mode.
    """

    with fake_mode(), track_torch_thunks() as hists:
        yield hists
