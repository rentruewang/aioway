# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn, optim

from aioway.attrs import Shape
from aioway.cases import ContrastiveLoss, ContrastiveLossEmitter
from aioway.emits import emit, linear_from_attr
from aioway.spaces import AttrSpace
from aioway.torch.nn import NnUFunc


@pytest.fixture
def contrastive():
    with ContrastiveLossEmitter(linear_from_attr).consider():
        yield


def test_emit_contrastive(contrastive):
    for out in emit(
        AttrSpace(shape=Shape.parse([3, 5, 7])),
        AttrSpace(shape=Shape.parse([3, 5, 7])),
    ):
        assert isinstance(out, NnUFunc)
        assert isinstance(out.module, ContrastiveLoss)


def test_contrastive_loss():
    loss = ContrastiveLoss(nn.Linear(3, 5), nn.MSELoss(), optim.Adam)

    tensor = torch.randn(7, 3).requires_grad_()
    out = loss(tensor)
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 0
