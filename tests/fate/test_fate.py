# Copyright (c) AIoWay Authors - All Rights Reserved

import operator

import pytest
import torch

from aioway.ctx import fake_fn


@pytest.fixture
def left(fake_mode):
    return torch.randn(3)


@pytest.fixture
def right(fake_mode):
    return torch.randn(3)


@pytest.fixture
def scalar():
    return 666


def _operators():
    yield operator.add
    yield operator.sub
    yield operator.mul
    yield operator.truediv
    yield operator.floordiv
    yield operator.pow
    yield operator.eq
    yield operator.ne
    yield operator.gt
    yield operator.ge
    yield operator.lt
    yield operator.le


@pytest.fixture(params=_operators())
def op(request: pytest.FixtureRequest):
    return request.param


def test_left_op_right(left, right, op):
    with fake_fn():
        result = op(left, right)
        assert isinstance(result, torch.Tensor)


def test_left_op_scalar(left, scalar, op):
    with fake_fn():
        result = op(left, scalar)
        assert isinstance(result, torch.Tensor)


def test_scalar_op_right(scalar, right, op):
    with fake_fn():
        result = op(scalar, right)
        assert isinstance(result, torch.Tensor)
