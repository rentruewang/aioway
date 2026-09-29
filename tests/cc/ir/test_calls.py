# Copyright (c) AIoWay Authors - All Rights Reserved

import torch

from aioway.cc import Dag, Exec, TorchFuncDag
from aioway.t import fake_mode


def test_torch_func_dag_data():
    a = torch.ones(3)
    b = torch.ones(3)
    dag = TorchFuncDag()

    with dag.activate():
        torch.add(a, b)

    assert len(dag.thunks) == 1
    assert dag.thunks[0].func is torch.add
    assert torch.equal(dag.thunks[0].result, torch.full((3,), 2.0))


def test_torch_func_dag_returns():
    a = torch.ones(3)
    b = torch.ones(3)
    dag = TorchFuncDag()

    with dag.activate():
        out = torch.add(a, b)

    assert torch.equal(out, torch.full((3,), 2.0))


def test_torch_func_calls_funcs():
    a = torch.ones(3)
    dag = TorchFuncDag()

    with dag.activate():
        torch.add(a, a)
        torch.mul(a, a)

    assert [t.func for t in dag.thunks] == [torch.add, torch.mul]


def test_seq_like_api():
    a = torch.ones(3)
    dag = TorchFuncDag()

    with dag.activate():
        torch.add(a, a)

    assert len(dag) == 1
    assert dag[0] is dag.thunks[0]


def test_dag_property():
    dag = TorchFuncDag()

    assert isinstance(dag.exec(), Exec)


def test_dag_from_trace():
    tracer = TorchFuncDag()

    with fake_mode():
        x, y = torch.zeros(3), torch.zeros(3)

        with tracer.activate():
            z = torch.add(x, y)

    dag = Dag.from_thunk_list(tracer.thunks)

    assert len(dag) == 1
    assert len(dag.inputs) == 2
    assert dag.inputs[0] is x
    assert dag.inputs[1] is y
    assert len(dag.outputs) == 1
    assert dag.outputs[0] is z
