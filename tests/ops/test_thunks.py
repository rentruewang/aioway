# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import operator
import typing
from collections import abc as cabc

import pytest
import torch
from torch import nn

from aioway._signs import Sign
from aioway.ops import CallExpr, ModuleExpr, TensorExpr, execute


@dcls.dataclass
class Case:
    name: str
    func: cabc.Callable
    args: tuple
    kwargs: dict
    expected: typing.Any

    def __init__(
        self, name: str, func: cabc.Callable, expected, *args, **kwargs
    ) -> None:
        self.name = name
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.expected = expected

        for arg in *self.args, *self.kwargs.values():
            # FIXME: Remove after domain tensor is done.
            if isinstance(arg, torch.Tensor):
                raise TypeError("Tensors should be wrapped.")


def _sub(a, b=0):
    return a - b


def call_cases():
    yield Case("add", operator.add, 3, 1, 2)
    yield Case("kwargs", _sub, 3, 5, b=2)
    yield Case(
        "torch_add",
        torch.add,
        torch.full((3,), 3.0),
        TensorExpr(torch.ones(3)),
        TensorExpr(torch.full((3,), 2.0)),
    )
    yield Case(
        "torch_relu",
        torch.relu,
        torch.tensor([0.0, 0.0, 2.0]),
        TensorExpr(torch.tensor([-1.0, 0.0, 2.0])),
    )
    yield Case(
        "torch_sum",
        torch.sum,
        torch.tensor([3.0, 12.0]),
        TensorExpr(torch.arange(6.0).reshape(2, 3)),
        dim=1,
    )


def module_cases():
    layer = nn.Linear(2, 3)
    x = torch.randn(1, 2)
    yield Case("linear", layer, layer(x), TensorExpr(x))


def all_cases():
    yield from call_cases()
    yield from module_cases()


def _case_id(case: Case) -> str:
    return case.name


@pytest.fixture(params=call_cases(), ids=_case_id)
def call_expr(request: pytest.FixtureRequest) -> CallExpr:
    case: Case = request.param
    return CallExpr(func=case.func, args=case.args, kwargs=case.kwargs)


@pytest.fixture(params=module_cases(), ids=_case_id)
def module_expr(request: pytest.FixtureRequest) -> ModuleExpr:
    case: Case = request.param
    assert isinstance(case.func, nn.Module)
    return ModuleExpr(func=case.func, args=case.args, kwargs=case.kwargs)


@pytest.fixture(params=all_cases(), ids=_case_id)
def any_case(request: pytest.FixtureRequest) -> Case:
    return request.param


@pytest.fixture
def any_expr(any_case: Case) -> CallExpr:
    cls: typing.Any = ModuleExpr if isinstance(any_case.func, nn.Module) else CallExpr
    return cls(func=any_case.func, args=any_case.args, kwargs=any_case.kwargs)


def test_call(any_expr: CallExpr, any_case: Case):
    torch.testing.assert_close(any_expr(), any_case.expected)


def test_module_call(module_expr: ModuleExpr):
    torch.testing.assert_close(
        module_expr(), module_expr.func(*[execute(a) for a in module_expr.args])
    )


def test_inputs_nested(call_expr: CallExpr):
    outer = CallExpr(func=lambda x: x, args=(call_expr,), kwargs={})

    found = list(outer.deps())
    assert len(found) == 1
    assert found[0] is call_expr


def test_sign(any_expr: CallExpr):
    assert isinstance(any_expr.sign, Sign)
