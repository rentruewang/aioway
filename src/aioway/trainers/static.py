# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

import torch
from torch import nn, optim
from torch.nn import utils as nn_utils

__all__ = [
    "LossFunc",
    "PredLossPair",
    "static_train_step",
    "static_infer_step",
]


class LossFunc(typing.Protocol):
    """
    A loss function should have the format (input, target) -> loss.
    """

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor: ...


class PredLossPair(typing.NamedTuple):
    "The prediction and loss pair."

    pred: torch.Tensor
    loss: torch.Tensor


def clip_gradients(params: cabc.Iterator[nn.Parameter], max_grad: float) -> None:
    if max_grad < 0:
        raise ValueError(f"{max_grad=} should be positive.")

    nn_utils.clip_grad_norm_(params, max_norm=max_grad)


def static_train_step(
    module: nn.Module,
    optimizer: optim.Optimizer,
    loss_func: LossFunc,
    x: torch.Tensor,
    y: torch.Tensor,
) -> PredLossPair:
    "Train a module in a static (non interactive) manner."

    inferred = static_infer_step(module, loss_func, x, y)

    optimizer.zero_grad()
    inferred.loss.backward()

    clip_gradients(module.parameters(), 1)

    optimizer.step()
    return inferred


def static_infer_step(
    module: nn.Module, loss_func: LossFunc, x: torch.Tensor, y: torch.Tensor
) -> PredLossPair:
    pred = module(x)
    loss = loss_func(pred, y)
    return PredLossPair(pred, loss)
