# Copyright (c) AIoWay Authors - All Rights Reserved

import operator
import pytest
import torch
from aioway.fake import is_fake_tensor
from aioway.fn import fake_fn


@pytest.fixture
def left():
    return torch.randn(3)


@pytest.fixture
def right():
    return torch.randn(3)


def _binary_ops():
    yield operator.add
    yield operator.sub
    yield operator.mul
    yield operator.truediv
    yield operator.floordiv
    yield operator.pow
    yield operator.eq
    yield operator.ne
    yield operator.lt
    yield operator.le
    yield operator.gt
    yield operator.ge


@pytest.fixture(params=_binary_ops())
def binary_op(request):
    return request.param


def test_binary_op(left, right, binary_op, maybe_fake_mode):
    result = binary_op(left, right)
    assert isinstance(result, torch.Tensor)
