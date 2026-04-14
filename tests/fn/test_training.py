# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn, optim
from torch.nn import functional as F


def _loss_fns():
    yield nn.L1Loss()
    yield F.l1_loss

    yield nn.MSELoss()
    yield F.mse_loss


@pytest.fixture
def input():
    return torch.randn(7, 3).requires_grad_()


@pytest.fixture
def target(input: torch.Tensor):
    return torch.randn_like(input)


@pytest.fixture(params=_loss_fns())
def loss(request: pytest.FixtureRequest, input: torch.Tensor, target: torch.Tensor):
    return request.param(input=input, target=target)


def _optimizer_types():
    yield optim.SGD
    yield optim.AdamW
    yield optim.Adam
    yield optim.RMSprop
    yield optim.NAdam


@pytest.fixture(params=_optimizer_types())
def optim_type(request: pytest.FixtureRequest):
    return request.param


def _lrs():
    yield 0.1
    yield 1e-3
    yield 2e-5


@pytest.fixture(params=_lrs())
def lr(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture
def optimizer(optim_type: type[optim.Optimizer], loss: torch.Tensor, lr: float):
    pytest.xfail("Optimizer not implemented yet.")
    return Optim(optim_cls=optim_type, params=loss.parameters(), lr=lr)


def test_loss_fn(loss: torch.Tensor):
    assert loss.shape.numel() == 1


def test_backward_fn(
    loss: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
):
    loss.backward()
    assert input.grad is not None
    assert target.grad is None


def test_optim_zero_grad(
    optimizer: optim.Optimizer,
    loss: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
):
    loss.backward()
    optimizer.zero_grad()
    assert input.grad is target.grad is None


def test_optim_step(
    optimizer: optim.Optimizer, loss: torch.Tensor, input: torch.Tensor
):
    original = input.clone()
    optimizer.zero_grad()
    assert input.grad is None
    loss.backward()
    assert input.grad is not None
    optimizer.step()

    # Test if optimization step did happen.
    updated = input != original
    assert updated.any()
