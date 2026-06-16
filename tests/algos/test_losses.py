# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway.algos import loss_func


@pytest.fixture
def input_mse():
    return torch.randn(3, 5)


@pytest.fixture
def target_mse():
    return torch.randn(3, 5)


def test_mse(input_mse: torch.Tensor, target_mse: torch.Tensor):
    l = loss_func(input_mse, target_mse)
    assert isinstance(l, torch.Tensor)
    assert l.ndim == 0


@pytest.fixture
def input_bce():
    return torch.rand([3, 5])


@pytest.fixture
def target_bce():
    return torch.rand([3, 5])


def test_bce(input_bce: torch.Tensor, target_bce: torch.Tensor):
    l = loss_func(input_bce, target_bce)
    assert isinstance(l, torch.Tensor)
    assert l.ndim == 0
