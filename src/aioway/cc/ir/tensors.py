# Copyright (c) AIoWay Authors - All Rights Reserved

"Tracking / routing related `Thunk`s."

import contextlib as ctxl
import dataclasses as dcls
import typing

from aioway.torch import (
    AtenThunk,
    TorchDispMode,
    TorchDispThunk,
    TorchFuncMode,
    TorchFuncThunk,
    fake_mode,
    route_aten_thunk,
)

from .hists import HistTensorGraph

__all__ = [
    "track_torch_thunks",
    "track_torch_fake_thunks",
    "TrackTorchDispHist",
    "TrackTorchFuncHist",
]


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
