# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

import pytest

from aioway.cc import Dag, DoneThunk


@pytest.fixture
def dag(thunks) -> Dag:
    return Dag(thunks)


@typing.no_type_check
def test_thunk_not_callable():
    with pytest.raises(TypeError):
        DoneThunk(func=3, args=(), kwargs={})


def test_thunk_upstream(thunks: cabc.Sequence[DoneThunk], fakes):
    x0, x1 = fakes
    assert list(thunks[0].upstreams) == [x0, x1]


def test_thunk_downstream(thunks: cabc.Sequence[DoneThunk]):
    assert list(thunks[0].downstreams) == [thunks[0].result]


def test_dag_len(dag: Dag):
    assert len(dag) == 2


def test_dag_getitem(dag: Dag, thunks):
    assert dag[0] is thunks[0]
    assert dag[-1] is thunks[1]


def test_dag_iter(dag: Dag, thunks):
    assert list(dag) == list(thunks)


def test_dag_slice(dag: Dag, thunks):
    sliced = dag[1:]

    assert isinstance(sliced, Dag)
    assert len(sliced) == 1
    assert sliced[0] is thunks[1]


def test_dag_inputs(dag: Dag, fakes):
    x0, x1 = fakes
    assert dag.inputs() == (x0, x1)


def test_dag_slice_inputs(dag: Dag, thunks):
    y = thunks[0].result
    assert dag[1:].inputs() == (y,)


def test_dag_var_list(dag: Dag, fakes, thunks):
    x0, x1 = fakes
    y, z = thunks[0].result, thunks[1].result
    var_list = dag.var_list()

    assert len(var_list) == 4
    assert var_list[x0].consumers == {0}
    assert var_list[y].producer == 0
    assert var_list[y].consumers == {1}
    assert var_list[z].is_output


def test_dag_output_unique(thunks):
    with pytest.raises(KeyError):
        Dag([thunks[0], thunks[0]])
