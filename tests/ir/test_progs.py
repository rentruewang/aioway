# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

import pytest

from aioway._pytrees import tree_flatten, tree_unflatten
from aioway.ir import FuncCall, Proc, TensorLifetime
from aioway.t import BoundArgs, TList


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
    def test_len(self, dag: Proc):
        assert len(dag) == 2

    def test_getitem(self, dag: Proc, thunks: cabc.Sequence[FuncCall]):
        assert dag[0] is thunks[0]
        assert dag[-1] is thunks[1]

    def test_iter(self, dag: Proc, thunks):
        assert list(dag) == list(thunks)

    def test_inputs(self, dag: Proc, fakes):
        assert set(dag.inputs.values()) == set(fakes)

    def test_outputs(self, dag: Proc, thunks):
        assert set(dag.outputs) == {thunks[-1].result}

    def test_output_unique(self, thunks, fakes):
        with pytest.raises(ValueError):
            Proc(
                instrs=[thunks[0], thunks[0]],
                inputs=BoundArgs.from_iters(fakes),
                outputs=TList.deduplicate([thunks[0].result]),
            )


class TestTensorIndex:
    def test_tensors(self, dag: Proc):
        assert len(dag.tindex.tensors) == 4

    def test_tensor_index_steps(self, dag: Proc, fakes, thunks):
        x0, _ = fakes
        y, z = thunks[0].result, thunks[1].result

        assert list(dag.tindex.input_to_step(x0)) == [0]

        assert dag.tindex.output_of_step(y) == 0
        assert list(dag.tindex.input_to_step(y)) == [1]
        assert dag.tindex.life(y) == TensorLifetime(0, 1)

        assert dag.tindex.output_of_step(z) == 1
        assert dag.tindex.life(z) == TensorLifetime(1, len(dag))


class TestProgramIn:
    def test_reconstruct(self, dag: Proc):
        data, spec = tree_flatten(dag.inputs)
        recon = tree_unflatten(data, spec)
        assert dag.inputs == recon
