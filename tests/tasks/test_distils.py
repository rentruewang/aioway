# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn, optim

from aioway.tasks.distils import distil


@pytest.fixture
def identity():
    return nn.Identity()


@pytest.fixture
def linear():
    return nn.Linear(3, 3, bias=False)


def _generate_batch():
    for _ in range(2000):
        yield torch.randn(1024, 3)


def test_distil(identity: nn.Module, linear: nn.Module):
    optimizer = optim.AdamW(linear.parameters(), lr=1e-2)

    for _ in distil(_generate_batch(), linear, identity, nn.L1Loss(), optimizer):
        pass

    weights = linear.weight
    assert isinstance(weights, torch.Tensor)
    assert torch.allclose(weights, torch.eye(3), atol=1e-2)
