# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

import pytest

from aioway.cc import Dag, TensorLifetime, ThunkNode


@typing.no_type_check
def test_thunk_not_callable():
    with pytest.raises(TypeError):
        ThunkNode(func=3, args=(), kwargs={}, result=None)


def test_thunk_upstream(thunks: cabc.Sequence[ThunkNode], fakes):
    assert set(thunks[0].inputs) == set(fakes)


def test_thunk_downstream(thunks: cabc.Sequence[ThunkNode]):
    assert list(thunks[0].outputs) == [thunks[0].result]


def test_dag_len(dag: Dag):
    assert len(dag) == 2


def test_dag_getitem(dag: Dag, thunks):
    assert dag[0] is thunks[0]
    assert dag[-1] is thunks[1]


def test_dag_iter(dag: Dag, thunks):
    assert list(dag) == list(thunks)


def test_dag_inputs(dag: Dag, fakes):
    assert set(dag.inputs) == set(fakes)


def test_dag_outputs(dag: Dag, thunks):
    assert set(dag.outputs) == {thunks[-1].result}


def test_dag_tensors(dag: Dag):
    assert len(dag.tensors) == 4


def test_dag_steps(dag: Dag, fakes, thunks):
    x0, _ = fakes
    y, z = thunks[0].result, thunks[1].result

    assert list(dag.input_to_step(x0)) == [0]

    assert dag.output_of_step(y) == 0
    assert list(dag.input_to_step(y)) == [1]
    assert dag.life(y) == TensorLifetime(0, 1)

    assert dag.output_of_step(z) == 1
    assert dag.life(z) == TensorLifetime(1, len(dag))


def test_dag_output_unique(thunks, fakes):
    with pytest.raises(ValueError):
        Dag([thunks[0], thunks[0]], inputs=fakes, outputs=[thunks[0].result])
