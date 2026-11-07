# Copyright (c) AIoWay Authors - All Rights Reserved

import operator
import typing

import pytest
import torch
from torch import nn

from aioway._signs import Sign
from aioway.exprs import CallExpr, ModuleExpr


class Case(typing.NamedTuple):
    func: typing.Callable
    args: tuple
    kwargs: dict
    expected: typing.Any


def _sub(a, b=0):
    return a - b


CASES = {
    "add": Case(operator.add, (1, 2), {}, 3),
    "kwargs": Case(_sub, (5,), {"b": 2}, 3),
    "torch_add": Case(
        torch.add,
        (torch.ones(3), torch.full((3,), 2.0)),
        {},
        torch.full((3,), 3.0),
    ),
    "torch_relu": Case(
        torch.relu,
        (torch.tensor([-1.0, 0.0, 2.0]),),
        {},
        torch.tensor([0.0, 0.0, 2.0]),
    ),
    "torch_sum": Case(
        torch.sum,
        (torch.arange(6.0).reshape(2, 3),),
        {"dim": 1},
        torch.tensor([3.0, 12.0]),
    ),
}


@pytest.fixture(params=list(CASES.values()), ids=list(CASES.keys()))
def case(request) -> Case:
    return request.param


@pytest.fixture
def call_expr(case) -> CallExpr:
    return CallExpr(func=case.func, args=case.args, kwargs=case.kwargs)


@pytest.fixture
def module_expr() -> ModuleExpr:
    torch.manual_seed(0)
    return ModuleExpr(func=nn.Linear(2, 3), args=(torch.randn(1, 2),), kwargs={})


@pytest.fixture(params=[call_expr.name, module_expr.name])
def any_expr(request):
    return request.getfixturevalue(request.param)


def test_call(call_expr, case):
    torch.testing.assert_close(call_expr(), case.expected)


def test_module_call(module_expr):
    out = module_expr()
    assert out.shape == (1, 3)
    torch.testing.assert_close(out, module_expr.func(*module_expr.args))


def test_inputs_empty(any_expr):
    assert list(any_expr.inputs()) == []


def test_inputs_nested(call_expr):
    outer = CallExpr(func=lambda x: x, args=(call_expr,), kwargs={})

    found = list(outer.inputs())
    assert len(found) == 1
    assert found[0] is call_expr


def test_sign(any_expr):
    assert isinstance(any_expr.sign, Sign)
