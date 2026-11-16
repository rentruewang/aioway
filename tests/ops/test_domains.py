# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn

from aioway.ops import (
    EmbeddingOp,
    GridOp,
    LayerOp,
    OpDomainMismatch,
    SameShapeOp,
    SoftmaxOp,
    ValueOp,
    ViewOp,
)


def _layers():
    yield LayerOp, nn.Linear(4, 3)
    yield SameShapeOp, nn.ReLU()
    yield ViewOp, nn.Flatten()
    yield EmbeddingOp, nn.Embedding(10, 4)
    yield SoftmaxOp, nn.Softmax(dim=1)
    yield GridOp, nn.Conv1d(3, 5, kernel_size=3)


@pytest.fixture(params=_layers(), ids=lambda layer: layer[0].__name__)
def layer(request: pytest.FixtureRequest):
    return request.param


def test_domain_check_tensor(layer):
    cls, module = layer

    # A tensor in `ValueOp` is still `AnyDomain`, the domain comes from the input `Op`.
    with pytest.raises(OpDomainMismatch):
        cls(module, ValueOp(torch.randn(8, 4)))
