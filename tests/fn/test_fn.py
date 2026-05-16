# Copyright (c) AIoWay Authors - All Rights Reserved


import pytest
import torch

from aioway.fate import Fate
from aioway.fn import (
    FateFn,
    Fn,
    MightFn,
    NnForwardFn,
    NnInitFn,
    TDispatchFn,
    TFunctionFn,
    TorchDispatchStack,
    TorchFunctionStack,
    track_fn,
)
from aioway.might import Might


def _fn_cls():
    yield FateFn
    yield TDispatchFn
    yield TFunctionFn
    yield MightFn
    yield NnInitFn
    yield NnForwardFn

    yield Fate
    yield Might


@pytest.fixture(params=_fn_cls())
def fn_cls(request):
    return request.param


@pytest.fixture
def a():
    return torch.randn(4)


@pytest.fixture
def b():
    return torch.randn(4)


def test_fn_subclass(fn_cls):
    assert isinstance(fn_cls, type)
    assert issubclass(fn_cls, Fn)


def test_call(a: torch.Tensor, b: torch.Tensor):
    with (
        track_fn() as [func_hist, dis_hist],
        TorchFunctionStack().ctx() as funcs,
        TorchDispatchStack().ctx() as ops,
    ):
        result = a + b

    assert result.ndim == 1
    assert len(func_hist)
    assert len(dis_hist)

    # After calling so of course it's empty
    assert not len(funcs.stack)
    assert not len(ops.stack)

    assert dis_hist.memory()
    assert str(dis_hist[0])
