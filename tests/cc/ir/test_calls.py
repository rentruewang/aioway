# Copyright (c) AIoWay Authors - All Rights Reserved

import torch
from torch import ops

from aioway.cc import Exec, TorchFuncDag, fake_aten_dag
from aioway.t import is_fake


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


def test_fake_aten_dag_tensor():
    with fake_aten_dag():
        out = torch.ones(3) + torch.ones(3)

    assert isinstance(out, torch.Tensor) and is_fake(out)


def test_fake_aten_dag_ops():
    with fake_aten_dag() as dag:
        torch.ones(3) + torch.ones(3)

    funcs = [t.func for t in dag.thunks]
    assert ops.aten.add.Tensor in funcs
