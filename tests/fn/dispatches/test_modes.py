# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway.fn import (
    fake_dispatch_fn,
    fake_mode,
    track_dispatch_fn,
)


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
    with track_dispatch_fn() as calls, function_fn_stack() as funcs:
        result = torch.einsum("i,j->", a, b)

    assert result.ndim == 0
    assert len(calls)
    assert len(funcs)
    assert calls.memory()


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
