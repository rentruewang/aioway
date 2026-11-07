# Copyright (c) AIoWay Authors - All Rights Reserved

import operator

import torch
from torch import nn

from aioway._signs import Sign
from aioway.exprs import CallExpr, ModuleExpr


def test_call_positional():
    expr = CallExpr(func=operator.add, args=(1, 2), kwargs={})
    assert expr() == 3


def test_call_kwargs():
    def f(a, b=0):
        return a - b

    expr = CallExpr(func=f, args=(5,), kwargs={"b": 2})
    assert expr() == 3


def test_torch_add():
    x, y = torch.ones(3), torch.full((3,), 2.0)
    expr = CallExpr(func=torch.add, args=(x, y), kwargs={})
    assert torch.equal(expr(), torch.full((3,), 3.0))


def test_torch_relu():
    x = torch.tensor([-1.0, 0.0, 2.0])
    expr = CallExpr(func=torch.relu, args=(x,), kwargs={})
    assert torch.equal(expr(), torch.tensor([0.0, 0.0, 2.0]))


def test_torch_sum_kwargs():
    x = torch.arange(6.0).reshape(2, 3)
    expr = CallExpr(func=torch.sum, args=(x,), kwargs={"dim": 1})
    assert torch.equal(expr(), torch.tensor([3.0, 12.0]))


def test_inputs_empty():
    expr = CallExpr(func=operator.add, args=(1, 2), kwargs={})
    assert list(expr.inputs()) == []


def test_inputs_nested():
    inner = CallExpr(func=operator.add, args=(1, 2), kwargs={})
    outer = CallExpr(func=operator.neg, args=(inner,), kwargs={})

    found = list(outer.inputs())
    assert len(found) == 1
    assert found[0] is inner


def test_inputs_ignores_tensors():
    expr = CallExpr(func=torch.relu, args=(torch.ones(2),), kwargs={})
    assert list(expr.inputs()) == []


def test_call_sign():
    expr = CallExpr(func=operator.add, args=(1, 2), kwargs={})
    assert isinstance(expr.sign, Sign)


def test_module_call():
    torch.manual_seed(0)
    layer = nn.Linear(2, 3)
    x = torch.randn(1, 2)

    expr = ModuleExpr(func=layer, args=(x,), kwargs={})
    out = expr()

    assert out.shape == (1, 3)
    assert torch.equal(out, layer(x))


def test_module_sign():
    expr = ModuleExpr(func=nn.Linear(2, 3), args=(torch.randn(1, 2),), kwargs={})
    assert isinstance(expr.sign, Sign)
