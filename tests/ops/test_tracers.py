# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
import torch
from torch import nn
from torch import testing as tt

from aioway.ops import (
    TensorOp,
    TorchFuncOpDag,
    ValueOp,
    fake_aten_op_dag,
    track_module_ops,
)
from aioway.ops._utils import exec_nested_ops
from aioway.t import fake_mode, is_aten_op


@pytest.fixture(params=["func", "aten"])
def tracer(request):
    return (
        TorchFuncOpDag().activate() if request.param == "func" else fake_aten_op_dag()
    )


@pytest.fixture(
    params=[
        lambda x, y: torch.mul(torch.add(x, y), x),
        lambda x, y: torch.sum(torch.stack([x, y]), 0),
        lambda x, y: torch.mul(torch.max(torch.stack([x, y]), dim=0).values, y),
    ]
)
def fn(request):
    return request.param


@pytest.fixture
def model():
    return nn.Sequential(nn.Linear(4, 8), nn.Sequential(nn.ReLU(), nn.Linear(8, 2)))


def trace(tracer, fn):
    with fake_mode():
        x, y = torch.zeros(3), torch.zeros(3)

        with tracer as dag:
            fn(x, y)

    return dag


def test_traced_matches_eager(tracer, fn):
    dag = trace(tracer, fn)
    x, y = torch.randn(3), torch.randn(3)

    tt.assert_close(dag.exec(x, y), fn(x, y))


def test_captures_input_tensors(tracer, fn):
    with fake_mode():
        x, y = torch.zeros(3), torch.zeros(3)

        with tracer as dag:
            fn(x, y)

    assert len(dag.inputs) == 2
    assert dag.inputs[0] is x and dag.inputs[1] is y


def test_aten_records_only_aten_ops(fn):
    dag = trace(fake_aten_op_dag(), fn)

    assert all(is_aten_op(thunk.func) for thunk in dag.history)


def test_module_traced_matches_eager(model):
    with track_module_ops() as hist:
        model(torch.zeros(3, 4))

    x = torch.randn(5, 4)
    tt.assert_close(hist.exec(x), model(x))


def test_module_records_only_leaves(model):
    with track_module_ops() as hist:
        model(torch.zeros(3, 4))

    leaves = [m for m in model.modules() if not list(m.children())]
    assert [thunk.func for thunk in hist.history] == leaves


class Pair(typing.NamedTuple):
    a: torch.Tensor
    b: torch.Tensor


class Split(nn.Module):
    def __init__(self, pack):
        super().__init__()
        self.pack = pack

    def forward(self, x):
        return self.pack(x * 2, x + 1)


class Add(nn.Module):
    def forward(self, a, b):
        return a + b


class SplitAdd(nn.Module):
    "A module returning a structure, consumed by the next module."

    def __init__(self, pack, unpack):
        super().__init__()
        self.split, self.add, self.unpack = Split(pack), Add(), unpack

    def forward(self, x):
        return self.add(*self.unpack(self.split(x)))


@pytest.mark.parametrize(
    "pack, unpack",
    [
        (lambda a, b: {"a": a, "b": b}, lambda out: (out["a"], out["b"])),
        (Pair, lambda out: (out.a, out.b)),
        (lambda a, b: ((a,), b), lambda out: (out[0][0], out[1])),
    ],
    ids=["dict", "namedtuple", "nested"],
)
def test_module_structured_outputs(pack, unpack):
    model = SplitAdd(pack, unpack)

    with track_module_ops() as hist:
        model(torch.zeros(3))

    x = torch.randn(3)
    tt.assert_close(hist.exec(x), model(x))


def test_exec_nested_ops_nested():
    tree = ([TensorOp(torch.ones(2)), 3], {"key": ValueOp("value")})

    [tensor, num], mapping = exec_nested_ops(tree)

    tt.assert_close(tensor, torch.ones(2))
    assert num == 3
    assert mapping == {"key": "value"}
