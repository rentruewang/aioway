# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

import pytest

from aioway.ir import FuncCall, Program, ProgramIn, TensorLifetime
from aioway.t import TList


class TestInstr:
    def test_not_callable(self) -> None:
        non_call: typing.Any = 3
        with pytest.raises(TypeError):
            FuncCall(func=non_call, args=(), kwargs={}, result=None)

    def test_upstream(self, thunks: cabc.Sequence[FuncCall], fakes):
        assert set(thunks[0].tensor_inputs) == set(fakes)

    def test_downstream(self, thunks: cabc.Sequence[FuncCall]):
        assert list(thunks[0].tensor_outputs) == [thunks[0].result]


class TestProgram:
    def test_len(self, dag: Program):
        assert len(dag) == 2

    def test_getitem(self, dag: Program, thunks: cabc.Sequence[FuncCall]):
        assert dag[0] is thunks[0]
        assert dag[-1] is thunks[1]

    def test_iter(self, dag: Program, thunks):
        assert list(dag) == list(thunks)

    def test_inputs(self, dag: Program, fakes):
        assert set(dag.inputs) == set(fakes)

    def test_outputs(self, dag: Program, thunks):
        assert set(dag.outputs) == {thunks[-1].result}

    def test_output_unique(self, thunks, fakes):
        with pytest.raises(ValueError):
            Program(
                instrs=[thunks[0], thunks[0]],
                inputs=ProgramIn.from_tensors(fakes),
                outputs=TList.from_iterable([thunks[0].result]),
            )


class TestTensorIndex:
    def test_tensors(self, dag: Program):
        assert len(dag.tindex.tensors) == 4

    def test_tensor_index_steps(self, dag: Program, fakes, thunks):
        x0, _ = fakes
        y, z = thunks[0].result, thunks[1].result

        assert list(dag.tindex.input_to_step(x0)) == [0]

        assert dag.tindex.output_of_step(y) == 0
        assert list(dag.tindex.input_to_step(y)) == [1]
        assert dag.tindex.life(y) == TensorLifetime(0, 1)

        assert dag.tindex.output_of_step(z) == 1
        assert dag.tindex.life(z) == TensorLifetime(1, len(dag))
