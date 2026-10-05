# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

import pytest

from aioway.ir import FCall, InstrSet, TensorLifetime


def test_thunk_not_callable() -> None:
    non_call: typing.Any = 3
    with pytest.raises(TypeError):
        FCall(func=non_call, args=(), kwargs={}, result=None)


def test_thunk_upstream(thunks: cabc.Sequence[FCall], fakes):
    assert set(thunks[0].inputs) == set(fakes)


def test_thunk_downstream(thunks: cabc.Sequence[FCall]):
    assert list(thunks[0].outputs) == [thunks[0].result]


def test_dag_len(dag: InstrSet):
    assert len(dag) == 2


def test_dag_getitem(dag: InstrSet, thunks: cabc.Sequence[FCall]):
    assert dag[0] is thunks[0]
    assert dag[-1] is thunks[1]


def test_dag_iter(dag: InstrSet, thunks):
    assert list(dag) == list(thunks)


def test_dag_inputs(dag: InstrSet, fakes):
    assert set(dag.inputs) == set(fakes)


def test_dag_outputs(dag: InstrSet, thunks):
    assert set(dag.outputs) == {thunks[-1].result}


def test_dag_tensors(dag: InstrSet):
    assert len(dag.tensors) == 4


def test_dag_steps(dag: InstrSet, fakes, thunks):
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
        InstrSet([thunks[0], thunks[0]], inputs=fakes, outputs=[thunks[0].result])
