# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn, optim

from aioway._specs import unbounded_box_spec
from aioway._torch import Shape
from aioway.cases import ContrastiveLoss, ContrastiveLossEmitter
from aioway.emits import emit_one, linear_shape


@pytest.fixture
def contrastive():
    with ContrastiveLossEmitter(linear_shape).consider():
        yield


def test_emit_contrastive(contrastive):
    out = emit_one(
        unbounded_box_spec(Shape.parse([3, 5, 7])),
        unbounded_box_spec(Shape.parse([3, 5, 7])),
    )

    assert isinstance(out, nn.Module)
    assert isinstance(out, ContrastiveLoss)


def test_contrastive_loss():
    loss = ContrastiveLoss(nn.Linear(3, 5), nn.MSELoss(), optim.Adam)

    tensor = torch.randn(7, 3).requires_grad_()
    out = loss(tensor)
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 0
