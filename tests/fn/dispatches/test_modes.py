# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway.fn import (
    fake_dispatch_fn,
    fake_mode,
    track_dispatch_fn,
)
from aioway.fn.routers import track_function_fn
from aioway.fn.tracking import TorchDispatchStack, TorchFunctionStack


@pytest.fixture
def a():
    return torch.randn(3)


@pytest.fixture
def b():
    return torch.randn(5)


@pytest.fixture
def fake_a():
    with fake_mode():
        return torch.randn(3)


@pytest.fixture
def fake_b():
    with fake_mode():
        return torch.randn(5)


def test_einsum(a: torch.Tensor, b: torch.Tensor):
    with (
        track_function_fn() as func_calls,
        track_dispatch_fn() as dis_calls,
        TorchFunctionStack() as funcs,
        TorchDispatchStack() as ops,
    ):
        result = torch.einsum("i,j->", a, b)

    assert result.ndim == 0
    assert len(func_calls)
    assert len(dis_calls)

    # Outside of calls, must be clear!
    assert not len(funcs.stack)
    assert not len(ops.stack)

    assert dis_calls.memory()


def test_boolean_masking_should_fail(fake_a: torch.Tensor):
    idx = torch.randn_like(fake_a) > 0

    with pytest.raises(RuntimeError):
        fake_a[idx]


def test_boolean_masking_patched(fake_a: torch.Tensor):
    idx = torch.randn_like(fake_a) > 0

    with fake_dispatch_fn():
        res = fake_a[idx]

    assert res.shape == fake_a.shape


def test_int_masking_ok(fake_a: torch.Tensor):
    idx = torch.randint(0, 1, [2])

    res = fake_a[idx]

    assert res.shape == (2,)
