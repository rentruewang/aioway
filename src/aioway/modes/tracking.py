# Copyright (c) AIoWay Authors - All Rights Reserved

"Tracking / routing related `Thunk`s."

import contextlib as ctxl
import dataclasses as dcls
import logging
import typing

import rich
from torch import nn

from aioway._utils import (
    current_fake_mode,
    replace_tensors,
    replace_tensors_with_attr,
    torch_fake_mode,
)

from .hists import Hist, HistTensorGraph
from .modules import NnFwdMode, NnFwdThunk, NnInitMode, NnInitThunk
from .tensors import TorchDispMode, TorchDispThunk, TorchFuncMode, TorchFuncThunk

if typing.TYPE_CHECKING:
    from aioway.modes import AtenThunk

__all__ = [
    "track_fn",
    "fake_fn",
    "PrintTorchFunc",
    "PrintTorchDisp",
    "LogTorchFunc",
    "LogTorchDis",
    "RouteTorchDisp",
    "RouteTorchFunc",
    "PrintNnInit",
    "PrintNnFwd",
]

LOGGER = logging.getLogger(__name__)


class _HasRichFlagMixin:
    def __init__(self, rich: bool = False) -> None:
        super().__init__()
        self._rich = rich


class PrintNnInit(NnInitMode):
    def run(self, thunk: NnInitThunk) -> nn.Module:
        print("invoke", thunk)
        result = thunk()
        print("return", thunk, "->", result)
        return result


class PrintNnFwd(NnFwdMode):
    def run(self, thunk: NnFwdThunk) -> object:
        print("invoke", thunk)
        result = thunk()
        print("return", thunk, "->", replace_tensors_with_attr(result))
        return result


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


class CloneDispatchOp(TorchDispMode):
    @typing.override
    def run(self, thunk: TorchDispThunk, /) -> object:
        result = thunk()

        # In fake mode, clone the tensor to prevent `FakeTensor` reuse. Should be cheap.
        if current_fake_mode():
            result = replace_tensors(result, lambda tensor: tensor.clone())

        return result


@dcls.dataclass
class RouteNnInit(NnInitMode):
    "The router at the `nn.Module` init level."

    history: Hist[NnInitThunk] = dcls.field(default_factory=Hist)
    """
    The history. Since it doesn't make sense to connect with `torch.Tensor`,
    we just use a plain `Hist` to store the history (no graph is needed).
    """

    @typing.override
    def run(self, thunk: NnInitThunk, /) -> nn.Module:
        result = self.history.execute(thunk)
        assert isinstance(result, nn.Module), type(result)
        return result


@dcls.dataclass
class RouteNnFwd(NnFwdMode):
    "The router at the `nn.Module` forward level."

    history: HistTensorGraph[NnFwdThunk] = dcls.field(default_factory=HistTensorGraph)
    """
    The history. Since `nn.Module`s in forward can be connected with `torch.Tensor`
    to for a computation graph on the module level, we use `HistTensorGraph`.
    """

    @typing.override
    def run(self, thunk: NnFwdThunk, /) -> object:
        return self.history.execute(thunk)


@dcls.dataclass
class RouteTorchDisp(TorchDispMode):
    "The router at the torch dispatch level."

    history: HistTensorGraph[TorchDispThunk | AtenThunk] = dcls.field(
        default_factory=HistTensorGraph
    )
    "The history used for tracking."

    def run(self, thunk: TorchDispThunk) -> object:
        from aioway.modes import AtenThunk

        fn: AtenThunk | TorchDispThunk

        if (found := AtenThunk.from_thunk(thunk)) is not None:
            fn = found

        # Cannot find corresponding operator, set it to the input `thunk`.
        else:
            fn = thunk

        assert isinstance(fn, TorchDispThunk | AtenThunk), type(fn)

        # Here, `AtenThunk` would do its magic and overwrite functions.
        return self.history.execute(fn)


@dcls.dataclass
class RouteTorchFunc(TorchFuncMode):
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
    nn_init: Hist[NnInitThunk]
    nn_fwd: Hist[NnFwdThunk]


@ctxl.contextmanager
def track_fn():
    """
    Track all calls into the torch dispatch mode as `TorchIrThunk`.
    """

    init = RouteNnInit()
    fwd = RouteNnFwd()
    dis = RouteTorchDisp()
    func = RouteTorchFunc()

    with func(), dis(), init(), fwd():
        yield HistoryCollection(
            function=func.history,
            dispatch=dis.history,
            nn_init=init.history,
            nn_fwd=fwd.history,
        )


@ctxl.contextmanager
def fake_fn():
    """
    Track all calls into the torch dispatch mode as `TorchIrThunk`,
    when fake mode is active.
    """

    with torch_fake_mode(), track_fn() as hists:
        yield hists
