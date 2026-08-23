# Copyright (c) AIoWay Authors - All Rights Reserved

"Tracking / routing related `Thunk`s."

import contextlib as ctxl
import dataclasses as dcls
import logging
import typing

import rich

from aioway._utils import replace_tensors
from aioway.schemas import replace_tensors_with_attr

from .fake import fake_mode, is_fake_mode_on
from .hists import HistTensorGraph
from .modes import TorchDispMode, TorchDispThunk, TorchFuncMode, TorchFuncThunk

if typing.TYPE_CHECKING:
    from aioway.modes import AtenThunk

__all__ = [
    "track_fn",
    "fake_fn",
    "PrintTorchFunc",
    "PrintTorchDisp",
    "LogTorchFunc",
    "LogTorchDis",
    "TrackTorchDispHist",
    "TrackTorchFuncHist",
    "RouteAtenThunkMode",
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


class CloneDispOp(TorchDispMode):
    """
    Automatically call `.clone()` on all tensors in the torch dispatch mode.

    This is useful to force a new `id` s.t. the tracking won't fail.
    """

    @typing.override
    def run(self, thunk: TorchDispThunk, /) -> object:
        result = thunk()

        # In fake mode, clone the tensor to prevent `FakeTensor` reuse. Should be cheap.
        if is_fake_mode_on():
            result = replace_tensors(result, lambda tensor: tensor.clone())

        return result


class RouteAtenThunkMode(TorchDispMode):
    """
    Route `torch.aten` calls to `AtenThunk` for some `aioway` specific functionalities.
    """

    def run(self, thunk: TorchDispThunk) -> object:
        from aioway.modes import AtenThunk

        fn: AtenThunk | TorchDispThunk

        if (found := AtenThunk.from_thunk(thunk)) is not None:
            fn = found

        # Cannot find corresponding operator, set it to the input `thunk`.
        else:
            fn = thunk

        assert isinstance(fn, TorchDispThunk | AtenThunk), type(fn)
        return fn()


@dcls.dataclass
class TrackTorchDispHist(TorchDispMode):
    "Track torch dispatch history."

    history: HistTensorGraph[TorchDispThunk | AtenThunk] = dcls.field(
        default_factory=HistTensorGraph
    )
    "The history used for tracking."

    def run(self, thunk: TorchDispThunk) -> object:
        # Here, `AtenThunk` would do its magic and overwrite functions.
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
def track_fn():
    """
    Track all calls into the torch dispatch mode as `TorchIrThunk`.
    """

    dis = TrackTorchDispHist()
    func = TrackTorchFuncHist()
    aten = RouteAtenThunkMode()

    with func.activate(), dis.activate(), aten.activate():
        yield HistoryCollection(function=func.history, dispatch=dis.history)


@ctxl.contextmanager
def fake_fn():
    """
    Track all calls into the torch dispatch mode as `TorchIrThunk`,
    when fake mode is active.
    """

    with fake_mode(), track_fn() as hists:
        yield hists
