# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn
from torch import testing as tt

from aioway.ops import (
    EmbeddingOp,
    GridOp,
    LayerOp,
    SameShapeOp,
    SoftmaxOp,
    TensorOp,
    ViewOp,
)


def _floats(*shape: int) -> TensorOp:
    return TensorOp(torch.randn(*shape))


def _ops():
    yield LayerOp(nn.Linear(4, 3), _floats(8, 4))
    yield SameShapeOp(nn.ReLU(), _floats(8, 4))
    yield ViewOp(nn.Flatten(), _floats(8, 2, 3))
    yield EmbeddingOp(nn.Embedding(10, 4), TensorOp(torch.randint(0, 10, (8,))))
    yield SoftmaxOp(nn.Softmax(dim=1), _floats(8, 4))
    yield GridOp(nn.Conv1d(3, 5, kernel_size=3), _floats(8, 3, 16))


@pytest.fixture(params=_ops(), ids=lambda op: type(op).__name__)
def op(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture
def softmax_op() -> SoftmaxOp:
    return SoftmaxOp(nn.Softmax(dim=1), _floats(8, 4))


def test_call(op):
    tt.assert_close(op(), op.module(op.input()))


def test_nn(op):
    assert op.net is op.module


def test_deps(op):
    assert op.__deps__ == (op.input,)


def test_softmax_dim(softmax_op: SoftmaxOp):
    assert softmax_op.dim == 1
