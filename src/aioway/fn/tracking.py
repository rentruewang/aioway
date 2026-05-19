# Copyright (c) AIoWay Authors - All Rights Reserved

"Tracking / routing related `Fn`s."

import contextlib as ctxl
import dataclasses as dcls
import logging
import typing

import rich
from torch import nn

from aioway.decomps import replace_tensors
from aioway.fake import enabled_fake_mode, torch_fake_mode

from .common import replace_tensors_with_attr
from .hists import Hist, HistTensorGraph
from .modules import NnFwdFn, NnFwdMode, NnInitFn, NnInitMode
from .tensors import TorDisFn, TorDisMode, TorFuncFn, TorFuncMode

if typing.TYPE_CHECKING:
    from aioway.fate import FateFn

__all__ = [
    "track_fn",
    "fake_fn",
    "PrintTorFunc",
    "PrintTorDis",
    "LogTorchFunc",
    "LogTorchDis",
    "RouteTorDis",
    "RouteTorFunc",
    "PrintNnInit",
    "PrintNnFwd",
]

LOGGER = logging.getLogger(__name__)


class _HasRichFlagMixin:
    def __init__(self, rich: bool = False) -> None:
        super().__init__()
        self._rich = rich


class PrintNnInit(NnInitMode):
    def __call__(self, thunk: NnInitFn) -> nn.Module:
        print("invoke", thunk)
        result = thunk.do()
        print("return", thunk, "->", result)
        return result


class PrintNnFwd(NnFwdMode):
    def __call__(self, thunk: NnFwdFn) -> object:
        print("invoke", thunk)
        result = thunk.do()
        print("return", thunk, "->", replace_tensors_with_attr(result))
        return result


class PrintTorFunc(_HasRichFlagMixin, TorFuncMode):
    @typing.override
    def __call__(self, thunk: TorFuncFn, /) -> object:
        return _TThunkPrinter(rich=self._rich)(thunk)


class PrintTorDis(_HasRichFlagMixin, TorDisMode):
    @typing.override
    def __call__(self, thunk: TorDisFn, /) -> object:
        return _TThunkPrinter(rich=self._rich)(thunk)


@dcls.dataclass(frozen=True)
class _TThunkPrinter:
    rich: bool
    "Use rich for printing."

    def __call__(self, thunk: TorFuncFn | TorDisFn) -> object:
        self.print("invoke", thunk)
        result = thunk.do()
        self.print("return", thunk, "->", replace_tensors_with_attr(result))
        return result

    @property
    def print(self):
        return rich.print if self.rich else print


@dcls.dataclass
class LogTorchFunc(TorFuncMode):
    """
    Log every call to function mode.
    """

    level: int
    "The level to log to."

    logger: logging.Logger = LOGGER
    "The logger to log to. Default to the one in the current module."

    @typing.override
    def __call__(self, thunk: TorFuncFn) -> object:
        result = thunk.do()
        self.logger.log(self.level, "%s", thunk)
        return result


@dcls.dataclass
class LogTorchDis(TorDisMode):
    """
    Log every call to dispatch mode.
    """

    level: int
    "The level to log to."

    logger: logging.Logger = LOGGER
    "The logger to log to. Default to the one in the current module."

    @typing.override
    def __call__(self, thunk: TorDisFn) -> object:
        result = thunk.do()
        self.logger.log(self.level, "%s", thunk)
        return result


class CloneDispatchOp(TorDisMode):
    @typing.override
    def __call__(self, thunk: TorDisFn, /) -> object:
        result = thunk.do()

        # In fake mode, clone the tensor to prevent `FakeTensor` reuse. Should be cheap.
        if enabled_fake_mode():
            result = replace_tensors(result, lambda tensor: tensor.clone())

        return result


@dcls.dataclass
class RouteNnInit(NnInitMode):
    "The router at the `nn.Module` init level."

    history: Hist[NnInitFn] = dcls.field(default_factory=Hist)
    """
    The history. Since it doesn't make sense to connect with `torch.Tensor`,
    we just use a plain `Hist` to store the history (no graph is needed).
    """

    @typing.override
    def __call__(self, thunk: NnInitFn, /) -> nn.Module:
        result = self.history.execute(thunk)
        assert isinstance(result, nn.Module), type(result)
        return result


@dcls.dataclass
class RouteNnFwd(NnFwdMode):
    "The router at the `nn.Module` forward level."

    history: HistTensorGraph[NnFwdFn] = dcls.field(default_factory=HistTensorGraph)
    """
    The history. Since `nn.Module`s in forward can be connected with `torch.Tensor`
    to for a computation graph on the module level, we use `HistTensorGraph`.
    """

    @typing.override
    def __call__(self, thunk: NnFwdFn, /) -> object:
        return self.history.execute(thunk)


@dcls.dataclass
class RouteTorDis(TorDisMode):
    "The router at the torch dispatch level."

    history: HistTensorGraph[TorDisFn | FateFn] = dcls.field(
        default_factory=HistTensorGraph
    )
    "The history used for tracking."

    def __call__(self, thunk: TorDisFn) -> object:
        from aioway.fate import FateFn

        fn: TorDisFn | FateFn

        if (fn := FateFn.find_fate(thunk)) is NotImplemented:
            # Cannot find corresponding operator, set it to the input `thunk`.
            fn = thunk

        assert isinstance(fn, TorDisFn | FateFn), type(fn)

        # Here, `FateFn` would do its magic and overwrite functions.
        return self.history.execute(fn)


@dcls.dataclass
class RouteTorFunc(TorFuncMode):
    """
    Saves the intermediate graph into a `FnHistory` object.
    """

    history: HistTensorGraph[TorFuncFn] = dcls.field(default_factory=HistTensorGraph)
    """
    The `HistTensorGraph` instance that would be responsible for tracking history,
    and which provides a graph API to interact with saved tensors.
    """

    @typing.override
    def __call__(self, thunk: TorFuncFn, /) -> object:
        return self.history.execute(thunk)


@ctxl.contextmanager
def track_fn():
    """
    Track all calls into the torch dispatch mode as `TorchIrFn`.
    """

    init = RouteNnInit()
    fwd = RouteNnFwd()
    dis = RouteTorDis()
    func = RouteTorFunc()

    with func.enter(), dis.enter(), init.enter(), fwd.enter():
        yield func.history, dis.history, init.history, fwd.history


@ctxl.contextmanager
def fake_fn():
    """
    Track all calls into the torch dispatch mode as `TorchIrFn`,
    when fake mode is active.
    """

    with torch_fake_mode(), track_fn() as hists:
        yield hists
