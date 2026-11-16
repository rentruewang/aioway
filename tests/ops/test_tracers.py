# Copyright (c) AIoWay Authors - All Rights Reserved

from collections import abc as cabc

import pytest
import torch
from torch import nn, ops
from torch import testing as tt

from aioway._utils import Stack
from aioway.ops import (
    CallOp,
    DagOpExec,
    LazyCacheOpExec,
    ModuleOpHist,
    ModuleOpTracker,
    ModuleThunkOp,
    OpFunc,
    OpHist,
    TensorOp,
    TorchFuncOpDag,
    fake_aten_op_dag,
    track_module_ops,
)
from aioway.t import fake_mode, has_fake, is_aten_op

type Fakes = tuple[torch.Tensor, torch.Tensor]


# Torch call tracers ====


def test_torch_func_dag_data():
    a = torch.ones(3)
    b = torch.ones(3)
    dag = TorchFuncOpDag()

    with dag.activate():
        torch.add(a, b)

    assert len(dag.history) == 1
    assert dag.history[0].func == torch.add
    assert torch.equal(dag.history[0](), torch.full((3,), 2.0))


def test_torch_func_dag_returns():
    a = torch.ones(3)
    b = torch.ones(3)
    dag = TorchFuncOpDag()

    with dag.activate():
        out = torch.add(a, b)

    assert torch.equal(out, torch.full((3,), 2.0))


def test_torch_func_calls_funcs():
    a = torch.ones(3)
    dag = TorchFuncOpDag()

    with dag.activate():
        torch.add(a, a)
        torch.mul(a, a)

    assert [t.func for t in dag.history] == [torch.add, torch.mul]


def test_seq_like_api():
    a = torch.ones(3)
    dag = TorchFuncOpDag()

    with dag.activate():
        torch.add(a, a)

    assert len(dag) == 1
    assert dag[0] is dag.history[0]


def test_dag_property():
    a = torch.ones(3)
    dag = TorchFuncOpDag()

    with dag.activate():
        torch.add(a, a)

    assert isinstance(dag.exec, OpFunc)


def test_empty_dag_no_output():
    with pytest.raises(ValueError):
        _ = OpHist().output


def test_fake_aten_dag_tensor():
    with fake_aten_op_dag():
        out = torch.ones(3) + torch.ones(3)

    assert isinstance(out, torch.Tensor) and has_fake(out)


def test_fake_aten_dag_ops():
    with fake_aten_op_dag() as dag:
        torch.ones(3) + torch.ones(3)

    funcs = [t.func for t in dag.history]
    assert ops.aten.add.Tensor in funcs


def test_dag_from_trace():
    tracer = TorchFuncOpDag()

    with fake_mode():
        x, y = torch.zeros(3), torch.zeros(3)

        with tracer.activate():
            z = torch.add(x, y)

    [add] = tracer.history
    left, right = add.args

    assert isinstance(left, TensorOp) and left.tensor is x
    assert isinstance(right, TensorOp) and right.tensor is y
    assert tracer.thunk_of(z) is add
    assert tracer.inputs == [x, y]


def test_dag_links_producers():
    tracer = TorchFuncOpDag()

    with fake_mode():
        x, y = torch.zeros(3), torch.zeros(3)

        with tracer.activate():
            torch.mul(torch.add(x, y), x)

    add, mul = tracer.history
    left, right = mul.args

    assert left is add
    assert right is add.args[0]


def test_dag_inputs_signature_order():
    tracer = TorchFuncOpDag()

    with fake_mode():
        x, y = torch.zeros(3), torch.zeros(3)

        # `y` is used first, but `OpFunc` binds in post order from the output.
        with tracer.activate():
            torch.add(x, torch.relu(y))

    assert tracer.inputs == [x, y]


def test_dag_inputs_many_same_step():
    tracer = TorchFuncOpDag()

    with fake_mode():
        xs = [torch.zeros(3) for _ in range(6)]

        with tracer.activate():
            torch.stack(xs)

    assert all(got is want for got, want in zip(tracer.inputs, xs, strict=True))


def test_dag_dead_inputs_not_in_signature():
    tracer = TorchFuncOpDag()

    with fake_mode():
        x, y = torch.zeros(3), torch.zeros(3)

        with tracer.activate():
            torch.relu(y)
            torch.neg(x)

    assert tracer.inputs == [x]


def trace_func(fn: cabc.Callable) -> tuple[OpFunc, Fakes]:
    tracer = TorchFuncOpDag()

    with fake_mode():
        x, y = torch.zeros(3), torch.zeros(3)

        with tracer.activate():
            fn(x, y)

    return tracer.exec, (x, y)


def trace_aten(fn: cabc.Callable) -> tuple[OpFunc, Fakes]:
    with fake_mode():
        x, y = torch.zeros(3), torch.zeros(3)

        with fake_aten_op_dag() as tracer:
            fn(x, y)

    return tracer.exec, (x, y)


def reals() -> Fakes:
    return torch.randn(3), torch.randn(3)


def _traces():
    yield pytest.param(trace_aten, id="aten")
    yield pytest.param(trace_func, id="func")


@pytest.fixture(params=_traces())
def trace(request) -> cabc.Callable[[cabc.Callable], tuple[OpFunc, Fakes]]:
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
    yield pytest.param(lambda x, y: torch.add(x, torch.relu(y)), id="late_input")
    yield pytest.param(lambda x, y: torch.sum(torch.stack([x, y]), 0), id="nested")
    yield pytest.param(
        lambda x, y: torch.mul(torch.max(torch.stack([x, y]), dim=0).values, y),
        id="tuple_result",
    )
    yield pytest.param(
        lambda x, y: torch.add(*torch.chunk(torch.cat([x, y]), 2)), id="chunks"
    )


@pytest.fixture(params=_fn())
def fn(request) -> cabc.Callable:
    return request.param


def test_traced_matches_eager(trace, fn):
    exec, _ = trace(fn)
    x, y = reals()

    tt.assert_close(exec(x, y), fn(x, y))


@pytest.mark.parametrize("exec_cls", [LazyCacheOpExec, DagOpExec])
def test_traced_matches_eager_cached(trace, fn, exec_cls):
    exec, _ = trace(fn)
    x, y = reals()

    with exec_cls.enable():
        tt.assert_close(exec(x, y), fn(x, y))


def test_run_twice_new_data(trace):
    fn = lambda x, y: torch.mul(torch.add(x, y), 2)
    exec, _ = trace(fn)

    for _ in range(2):
        x, y = reals()
        tt.assert_close(exec(x, y), fn(x, y))


def test_inputs_are_traced_params(trace):
    exec, _ = trace(lambda x, y: torch.add(x, y))

    assert list(exec.signature.params) == ["t0", "t1"]


def test_aten_records_only_aten_ops():
    with fake_mode():
        x, y = torch.zeros(3), torch.zeros(3)

        with fake_aten_op_dag() as tracer:
            torch.mul(torch.add(x, y), 2)

    assert len(tracer) == 2
    assert all(is_aten_op(thunk.func) for thunk in tracer.history)


# Module tracer ====


class Double(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


class Pair(nn.Module):
    """Two inputs, two outputs: passes each argument straight through."""

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x, y


class Wrapped(nn.Module):
    """A parent whose output is a fresh tensor, not its child's."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = Double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x) + 1


def _append_thunk(
    hist: ModuleOpHist, module: nn.Module, *args: torch.Tensor
) -> ModuleThunkOp:
    thunk = hist.thunk(ModuleThunkOp, module, args, {})
    hist.append(thunk, module(*args))
    return thunk


@pytest.fixture
def module() -> nn.Module:
    return Double()


@pytest.fixture
def hist() -> ModuleOpHist:
    return ModuleOpHist()


@pytest.fixture
def tracker(hist: ModuleOpHist) -> ModuleOpTracker:
    return ModuleOpTracker(stack=Stack(), hist=hist)


def test_thunk_upstream_yields(hist: ModuleOpHist) -> None:
    x, y = torch.ones(3), torch.zeros(3)
    thunk = _append_thunk(hist, Pair(), x, y)

    assert [arg.tensor for arg in thunk.args] == [x, y]


def test_hist_starts_empty(hist: ModuleOpHist) -> None:
    assert len(hist) == 0


def test_append_stores_thunk(hist: ModuleOpHist, module: nn.Module) -> None:
    thunk = _append_thunk(hist, module, torch.ones(3))

    assert len(hist) == 1
    assert hist[0] is thunk


def test_thunk_looks_up(hist: ModuleOpHist, module: nn.Module) -> None:
    x = torch.ones(3)
    y = module(x)
    thunk = hist.thunk(ModuleThunkOp, module, (x,), {})

    hist.append(thunk, y)

    assert hist.thunk_of(y) is thunk


def test_append_index_all_outputs(hist: ModuleOpHist) -> None:
    x, y = torch.ones(3), torch.zeros(3)
    thunk = _append_thunk(hist, Pair(), x, y)

    for idx, out in enumerate([x, y]):
        getter = hist.thunk_of(out)
        assert isinstance(getter, CallOp)
        assert getter.args == (thunk, idx)


def test_append_fail_dups(hist: ModuleOpHist, module: nn.Module) -> None:
    x = torch.ones(3)
    y = module(x)
    thunk = hist.thunk(ModuleThunkOp, module, (x,), {})
    hist.append(thunk, y)

    with pytest.raises(KeyError):
        hist.append(thunk, y)


def test_thunk_of_unknown_fail(hist: ModuleOpHist) -> None:
    with pytest.raises(KeyError):
        hist.thunk_of(torch.ones(3))


def test_hook_appends_thunk(hist: ModuleOpHist, module: nn.Module) -> None:
    x = torch.ones(3)
    y = module(x)

    hist.module_forward_hook(module, (x,), y)

    (recorded,) = hist.history
    assert recorded.func is module
    assert [arg.tensor for arg in recorded.args] == [x]
    assert hist.thunk_of(y) is recorded


def test_tracker_yield(tracker: ModuleOpTracker) -> None:
    with tracker() as yielded:
        assert yielded is tracker


def test_tracker_records_leaves(tracker: ModuleOpTracker, hist: ModuleOpHist) -> None:
    model = Wrapped()

    with tracker():
        model(torch.ones(3))

    assert [t.func for t in hist.history] == [model.inner]


def test_tracker_links_modules(tracker: ModuleOpTracker, hist: ModuleOpHist) -> None:
    model = nn.Sequential(nn.Linear(4, 2), nn.ReLU())

    with tracker():
        model(torch.ones(3, 4))

    linear, relu = hist.history
    assert relu.args == (linear,)


def test_tracker_pass_through(tracker: ModuleOpTracker, hist: ModuleOpHist) -> None:
    pair, double = Pair(), Double()

    with tracker():
        _, out = pair(torch.ones(3), torch.zeros(3))
        double(out)

    recorded_pair, recorded_double = hist.history
    assert recorded_double.args == (hist.thunk_of(out),)
    thunk = hist.thunk_of(out)
    assert isinstance(thunk, CallOp)
    assert thunk.args == (recorded_pair, 1)


def test_tracker_empty_stack(tracker: ModuleOpTracker) -> None:
    with tracker():
        Wrapped()(torch.ones(3))

    assert not tracker.stack


def test_tracker_active_only_context(
    tracker: ModuleOpTracker, hist: ModuleOpHist
) -> None:
    with tracker():
        pass

    Wrapped()(torch.ones(3))

    assert len(hist) == 0


def test_module_traced_matches_eager() -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    with track_module_ops() as hist:
        model(torch.zeros(3, 4))

    x = torch.randn(5, 4)
    tt.assert_close(hist.exec(x), model(x))


def test_module_exec_exposes_parameters() -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    with track_module_ops() as hist:
        model(torch.zeros(3, 4))

    got = {id(p) for p in hist.exec.module.parameters()}
    assert got == {id(p) for p in model.parameters()}
