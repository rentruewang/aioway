# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import contextlib as ctxl
import dataclasses as dcls
import typing

from aioway.torch import (
    TorchDispMode,
    TorchFuncMode,
    fake_mode,
    is_aten_op,
    route_aten_thunk,
)
from aioway.torch.guards import is_torch_function

from .dags import Dag, DoneThunk

__all__ = ["TorchFuncDag", "fake_aten_dag"]


@dcls.dataclass
class _TorchCallDag(abc.ABC):
    "The base class for torch functions or dispatches."

    thunks: list[DoneThunk] = dcls.field(default_factory=list)
    "The thunk storage."

    def __len__(self) -> int:
        return len(self.thunks)

    def __getitem__(self, idx: int) -> DoneThunk:
        return self.thunks[idx]

    def append(self, thunk: DoneThunk) -> None:
        self.thunks.append(thunk)

    @property
    def dag(self) -> Dag:
        return Dag(self.thunks)

    def run(self, thunk):
        result = thunk()

        # Only track the function if it is point of interest.
        if self._track_thunk(thunk.func):
            thunk_with_output = DoneThunk(
                func=thunk.func, args=thunk.args, kwargs=thunk.kwargs, result=result
            )
            self.thunks.append(thunk_with_output)

        return result

    @abc.abstractmethod
    def _track_thunk(self, func) -> bool:
        raise NotImplementedError


@typing.final
@dcls.dataclass
class TorchFuncDag(_TorchCallDag, TorchFuncMode):
    """
    The DAG for `__torch_function__` calls.
    """

    def _track_thunk(self, func) -> bool:
        return is_torch_function(func)


@typing.final
@dcls.dataclass
class _AtenDag(_TorchCallDag, TorchDispMode):
    """
    The DAG for `__torch_dispatch__` calls.

    Will only track `aten` ops, others are discarded.
    """

    def _track_thunk(self, func) -> bool:
        return is_aten_op(func)


@ctxl.contextmanager
def fake_aten_dag():
    """
    Test the DAG made from aten overrides.
    """

    dag = _AtenDag()

    with fake_mode(), route_aten_thunk.activate(), dag.activate():
        yield dag
