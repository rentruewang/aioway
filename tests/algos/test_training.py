# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

import pytest
import torch
from torch import nn, optim
from torch.nn import functional as F

from aioway.modes import track_fn


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


class LossParam(typing.NamedTuple):
    loss: torch.Tensor
    params: list[torch.Tensor]


@pytest.fixture(params=_loss_fns())
def loss_params(
    request: pytest.FixtureRequest, input: torch.Tensor, target: torch.Tensor
):
    with track_fn() as [_, tracker, _, _]:
        loss = request.param(input=input, target=target)
    params = list(tracker.parameters())
    return LossParam(loss=loss, params=params)


@pytest.fixture
def loss(loss_params: LossParam):
    return loss_params.loss


@pytest.fixture
def params(loss_params: LossParam):
    return loss_params.params


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
def optimizer(
    optim_type: cabc.Callable[..., optim.Optimizer], loss_params: LossParam, lr: float
):
    return optim_type(params=loss_params.params, lr=lr)


def test_params(params: list[torch.Tensor]):
    assert len(params)


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
