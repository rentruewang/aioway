# Copyright (c) AIoWay Authors - All Rights Reserved

from collections import abc as cabc

import pytest
import torch
from torch import ops
from torch import testing as tt

from aioway.ir import Exec, Program, TorchFuncDag, fake_aten_dag
from aioway.t import fake_mode, has_fake, is_aten_op, parse_attr

type Fakes = tuple[torch.Tensor, torch.Tensor]


def test_torch_func_dag_data():
    a = torch.ones(3)
    b = torch.ones(3)
    dag = TorchFuncDag()

    with dag.activate():
        torch.add(a, b)

    assert len(dag.thunks) == 1
    assert dag.thunks[0].func == torch.add
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

    assert isinstance(out, torch.Tensor) and has_fake(out)


def test_fake_aten_dag_ops():
    with fake_aten_dag() as dag:
        torch.ones(3) + torch.ones(3)

    funcs = [t.func for t in dag.thunks]
    assert ops.aten.add.Tensor in funcs


def test_dag_from_trace():
    tracer = TorchFuncDag()

    with fake_mode():
        x, y = torch.zeros(3), torch.zeros(3)

        with tracer.activate():
            z = torch.add(x, y)

    dag = Program.from_thunk_list(tracer.thunks)

    assert len(dag) == 1
    assert len(dag.inputs) == 2
    dix, diy = dag.inputs
    assert _attr_eq(dix, x)
    assert _attr_eq(diy, y)

    assert len(dag.outputs) == 1
    [doz] = dag.outputs
    assert _attr_eq(doz, z)


def test_dag_inputs_first_use_order():
    tracer = TorchFuncDag()

    with fake_mode():
        x, y = torch.zeros(3), torch.zeros(3)

        with tracer.activate():
            torch.add(torch.mul(x, y), x)

    dag = Program.from_thunk_list(tracer.thunks)

    assert _attr_eq(dag.inputs[0], x)
    assert _attr_eq(dag.inputs[1], y)


def test_dag_inputs_many_same_step():
    tracer = TorchFuncDag()

    with fake_mode():
        xs = [torch.zeros(3) for _ in range(6)]

        with tracer.activate():
            torch.stack(xs)

    dag = Program.from_thunk_list(tracer.thunks)

    assert all(_attr_eq(got, want) for got, want in zip(dag.inputs, xs))


def trace_func(fn: cabc.Callable) -> tuple[Exec, Fakes]:
    tracer = TorchFuncDag()

    with fake_mode():
        x, y = torch.zeros(3), torch.zeros(3)

        with tracer.activate():
            fn(x, y)

    return tracer.exec(), (x, y)


def trace_aten(fn: cabc.Callable) -> tuple[Exec, Fakes]:
    with fake_mode():
        x, y = torch.zeros(3), torch.zeros(3)

        with fake_aten_dag() as tracer:
            fn(x, y)

    return tracer.exec(), (x, y)


def reals() -> Fakes:
    return torch.randn(3), torch.randn(3)


def _traces():
    yield pytest.param(trace_aten, id="aten")
    yield pytest.param(trace_func, id="func")


@pytest.fixture(params=_traces())
def trace(request) -> cabc.Callable[[cabc.Callable], tuple[Exec, Fakes]]:
    return request.param


def _fn():
    yield pytest.param(lambda x, y: torch.add(x, y), id="add")
    yield pytest.param(lambda x, y: torch.mul(torch.add(x, y), 2), id="add_scale")
    yield pytest.param(lambda x, y: torch.relu(torch.sub(x, y)), id="sub_relu")
    yield pytest.param(lambda x, y: torch.add(torch.mul(x, y), x), id="reuse_input")
    yield pytest.param(
        lambda x, y: torch.mul(torch.add(x, y), torch.sum(torch.add(x, y))),
        id="reuse_step",
    )


@pytest.fixture(params=_fn())
def fn(request) -> cabc.Callable:
    return request.param


def test_traced_matches_eager(trace, fn):
    exec, _ = trace(fn)
    x, y = reals()

    tt.assert_close(exec(x, y), fn(x, y))


def test_run_twice_new_data(trace):
    fn = lambda x, y: torch.mul(torch.add(x, y), 2)
    exec, _ = trace(fn)

    for _ in range(2):
        x, y = reals()
        tt.assert_close(exec(x, y), fn(x, y))


def test_inputs_are_traced_fakes(trace):
    exec, (x, y) = trace(lambda x, y: torch.add(x, y))

    assert len(exec.inputs) == 2

    fx, fy = exec.inputs
    assert _attr_eq(fx, x)
    assert _attr_eq(fy, y)


def test_wrong_shape_rejected(trace):
    exec, _ = trace(lambda x, y: torch.add(x, y))

    with pytest.raises(TypeError):
        exec(torch.randn(4), torch.randn(4))


def test_aten_records_only_aten_ops():
    exec, _ = trace_aten(lambda x, y: torch.mul(torch.add(x, y), 2))

    assert len(exec) == 2
    assert all(is_aten_op(thunk.func) for thunk in exec)


def _attr_eq(left: torch.Tensor, right: torch.Tensor):
    return parse_attr(left) == parse_attr(right)
