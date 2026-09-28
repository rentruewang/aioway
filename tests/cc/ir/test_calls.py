import torch
from torch import ops
from torch._subclasses.fake_tensor import FakeTensor

from aioway.cc import Dag, TorchFuncDag, fake_aten_dag


def test_torch_func_dag_records_a_call():
    a = torch.ones(3)
    b = torch.ones(3)
    dag = TorchFuncDag()

    with dag.activate():
        torch.add(a, b)

    assert len(dag.thunks) == 1
    assert dag.thunks[0].func is torch.add
    assert torch.equal(dag.thunks[0].result, torch.full((3,), 2.0))


def test_torch_func_dag_returns_the_real_result():
    a = torch.ones(3)
    b = torch.ones(3)
    dag = TorchFuncDag()

    with dag.activate():
        out = torch.add(a, b)

    assert torch.equal(out, torch.full((3,), 2.0))


def test_torch_func_dag_records_calls_in_order():
    a = torch.ones(3)
    dag = TorchFuncDag()

    with dag.activate():
        torch.add(a, a)
        torch.mul(a, a)

    assert [t.func for t in dag.thunks] == [torch.add, torch.mul]


def test_indexing_and_len():
    a = torch.ones(3)
    dag = TorchFuncDag()

    with dag.activate():
        torch.add(a, a)

    assert len(dag) == 1
    assert dag[0] is dag.thunks[0]


def test_dag_property_returns_a_dag():
    dag = TorchFuncDag()

    assert isinstance(dag.dag, Dag)


def test_fake_aten_dag_produces_fake_tensors():
    with fake_aten_dag():
        out = torch.ones(3) + torch.ones(3)

    assert isinstance(out, FakeTensor)


def test_fake_aten_dag_records_ops():
    with fake_aten_dag() as dag:
        torch.ones(3) + torch.ones(3)

    funcs = [t.func for t in dag.thunks]
    assert ops.aten.add.Tensor in funcs
