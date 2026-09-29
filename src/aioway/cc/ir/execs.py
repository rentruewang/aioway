# Copyright (c) AIoWay Authors - All Rights Reserved

"Executing the DAG with new data."

import typing
from collections import abc as cabc

import torch

from .dags import Dag, ThunkNode
from .vars import LocalScope

__all__ = ["Exec"]


class Exec[F: cabc.Callable = typing.Any](cabc.Sequence[ThunkNode[F]]):
    """
    This is the DAG executor responsible for executing a traced thunk list on real data.
    """

    def __init__(self, dag: Dag) -> None:
        self._dag = dag
        self._scope = LocalScope(self._dag.var_list())

    def __len__(self) -> int:
        return len(self._dag)

    @typing.overload
    def __getitem__(self, idx: int) -> ThunkNode[F]: ...

    @typing.overload
    def __getitem__(self, idx: slice) -> typing.Self: ...

    def __getitem__(self, idx):
        match idx:
            case int():
                return self._dag[idx]
            case slice():
                return type(self)(self._dag[idx])

    def __iter__(self) -> cabc.Generator[ThunkNode[F]]:
        yield from self._dag

    def __call__(self, *inputs: torch.Tensor) -> typing.Any:
        try:
            self._scope.update(self.inputs(), inputs)
        except ValueError as err:
            raise TypeError from err

        # Execute the steps one by one in topo sorted order.
        for idx, thunk in enumerate(self._dag):
            args, kwargs = self._scope.map([thunk.args, thunk.kwargs])
            real = thunk.func(*args, **kwargs)
            self._scope.update(thunk.result, real)
            self._scope.expire(idx)

        result = self._scope.map(self._dag[-1].result)
        self._scope.clear()
        return result

    def inputs(self):
        return self._dag.inputs()
