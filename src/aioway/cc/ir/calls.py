# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import contextlib as ctxl
import dataclasses as dcls
import typing

from aioway.torch import (
    TorchDispMode,
    TorchDispThunk,
    TorchFuncMode,
    TorchFuncThunk,
    fake_mode,
    is_aten_op,
)

from .dags import Dag, DoneThunk

__all__ = ["TorchFuncDag", "AtenDag", "fake_aten_dag"]


@dcls.dataclass
class _TorchCallDag(abc.ABC):
    "The base class for torch functions or dispatches."

    thunks: list[DoneThunk] = dcls.field(default_factory=list)
    "The thunk storage."

    def __len__(self) -> int:
        return len(self)

    def __getitem__(self, idx: int) -> DoneThunk:
        return self.thunks[idx]

    def append(self, thunk: DoneThunk) -> None:
        self.thunks.append(thunk)

    @property
    def dag(self) -> Dag:
        return Dag(self.thunks)


@typing.final
@dcls.dataclass
class TorchFuncDag(_TorchCallDag, TorchFuncMode):
    """
    The DAG for `__torch_function__` calls.
    """

    @typing.override
    def run(self, thunk: TorchFuncThunk):
        result = thunk()

        self.thunks.append(
            DoneThunk(
                func=thunk.func, args=thunk.args, kwargs=thunk.kwargs, result=result
            )
        )

        return result


@typing.final
@dcls.dataclass
class AtenDag(_TorchCallDag, TorchDispMode):
    """
    The DAG for `__torch_dispatch__` calls.

    Will only track `aten` ops, others are discarded.
    """

    @typing.override
    def append(self, thunk: DoneThunk) -> None:
        if not is_aten_op(thunk.func):
            return

        super().append(thunk)

    @typing.override
    def run(self, thunk: TorchDispThunk):
        result = thunk()

        self.thunks.append(
            DoneThunk(
                func=thunk.func, args=thunk.args, kwargs=thunk.kwargs, result=result
            )
        )

        return result


@ctxl.contextmanager
def fake_aten_dag():
    dag = AtenDag()

    with fake_mode(), dag.activate():
        yield dag
