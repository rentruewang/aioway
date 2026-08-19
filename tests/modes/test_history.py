# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway._torch import fake_mode, is_fake_mode_on
from aioway.modes import fake_fn, track_fn


@pytest.fixture
def a():
    return torch.randn(3)


@pytest.fixture
def b():
    return torch.randn(5)


@pytest.fixture
def c():
    return torch.randn(3)


@pytest.fixture
def d():
    return torch.randn(3, 11)


def test_fake_mode(fake_mode):
    assert is_fake_mode_on()


def test_real_mode(real_mode):
    assert not is_fake_mode_on()


@pytest.fixture
def fake_a():
    with fake_mode():
        return torch.randn(3)


@pytest.fixture
def fake_b():
    with fake_mode():
        return torch.randn(5)


def test_einsum(a: torch.Tensor, b: torch.Tensor):
    with track_fn() as [func_calls, dis_calls]:
        result = torch.einsum("i,j->", a, b)

    assert result.ndim == 0
    assert len(func_calls)
    assert len(dis_calls)

    assert dis_calls.memory()


def test_call(a: torch.Tensor, c: torch.Tensor):
    with track_fn() as [func_hist, dis_hist]:
        result = a + c

    assert result.ndim == 1
    assert func_hist
    assert dis_hist

    assert dis_hist.memory()
    assert str(dis_hist[0])


def test_boolean_masking_should_fail(fake_a: torch.Tensor):
    idx = torch.randn_like(fake_a) > 0

    with pytest.raises(RuntimeError):
        fake_a[idx]


def test_boolean_masking_patched(fake_a: torch.Tensor):
    idx = torch.randn_like(fake_a) > 0

    with fake_fn():
        res = fake_a[idx]

    assert res.shape == fake_a.shape


def test_int_masking_ok(fake_a: torch.Tensor):
    idx = torch.randint(0, 1, [2])

    res = fake_a[idx]

    assert res.shape == (2,)
