# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn

from aioway._utils import current_fake_mode, torch_fake_mode
from aioway.modes import NnFwdFn, NnInitFn, fake_fn, track_fn


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
    assert current_fake_mode()


def test_real_mode(real_mode):
    assert not current_fake_mode()


@pytest.fixture
def fake_a():
    with torch_fake_mode():
        return torch.randn(3)


@pytest.fixture
def fake_b():
    with torch_fake_mode():
        return torch.randn(5)


def test_einsum(a: torch.Tensor, b: torch.Tensor):
    with track_fn() as [func_calls, dis_calls, init_calls, fwd_calls]:
        result = torch.einsum("i,j->", a, b)

    assert result.ndim == 0
    assert len(func_calls)
    assert len(dis_calls)

    assert dis_calls.memory()

    assert not len(init_calls)
    assert not len(fwd_calls)


def test_call(a: torch.Tensor, c: torch.Tensor):
    with track_fn() as [func_hist, dis_hist, init_calls, fwd_calls]:
        result = a + c

    assert result.ndim == 1
    assert func_hist
    assert dis_hist

    assert dis_hist.memory()
    assert str(dis_hist[0])

    assert not init_calls
    assert not fwd_calls


def test_module_init():
    with track_fn() as [func_hist, dis_hist, init_calls, fwd_calls]:
        m = NnInitFn(nn.Linear, 11, 13)()

    assert isinstance(m, nn.Module)
    assert init_calls
    assert not fwd_calls


def test_module_fwd(d: torch.Tensor):
    m = NnInitFn(nn.Linear, 11, 13)()

    with track_fn() as [func_hist, dis_hist, init_calls, fwd_calls]:
        o = NnFwdFn(m, d)()

    assert isinstance(o, torch.Tensor)
    assert o.shape == (3, 13)

    assert isinstance(m, nn.Module)
    assert not init_calls
    assert fwd_calls


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
