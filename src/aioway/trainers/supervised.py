# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

import tensordict as td
import torch
from rich import progress
from torch import nn, optim
from torch.nn import utils as nn_utils
from torch.utils import data as dutils

from aioway.errors import re_raise_func

__all__ = [
    "VectorPair",
    "SupervisedTrainer",
    "ComputeSlLoss",
    "optimize",
    "optimize_clip",
    "static_train",
    "static_infer",
]


class LossFunc(typing.Protocol):
    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor: ...


def clip_gradients(params: cabc.Iterator[nn.Parameter], max_grad: float) -> None:
    if max_grad < 0:
        raise ValueError(f"{max_grad=} should be positive.")

    nn_utils.clip_grad_norm_(params, max_norm=max_grad)


def static_train(
    module: nn.Module,
    optimizer: optim.Optimizer,
    loss_func: LossFunc,
    loader: dutils.DataLoader,
):
    "Train a module in a static (non interactive) manner."

    for x, y in progress.track(loader):
        pred, loss = static_infer(module, loss_func, x, y)

        optimizer.zero_grad()
        loss.backward()
        clip_gradients(module.parameters(), 1)
        optimizer.step()
        print(loss)


def static_infer(
    module: nn.Module, loss_func: LossFunc, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = module(x)
    loss = loss_func(pred, y)
    return pred, loss


class VectorPair(td.TensorClass):
    "Supervised vector from input to output."

    x: torch.Tensor
    "The x tensor. Of shape [batch, in_feats]."

    y: torch.Tensor
    "The y tensor. Of shape [batch, out_feats]."

    @re_raise_func(AssertionError, ValueError)
    def __post_init__(self) -> None:
        assert self.x.ndim == self.y.ndim
        assert len(self.x) == len(self.y)
        self.auto_batch_size_()


@dcls.dataclass
class SupervisedTrainer:
    compute_loss: nn.Module
    "The module to use."

    optimizer: optim.Optimizer
    "The optimizer that optimizes the module."

    dataloader: dutils.DataLoader[VectorPair]
    "The training dataloader with the pairs of data."

    max_grad_norm: float | None = None
    "If specified, the maximum gradient norm."

    def train_epoch(self) -> cabc.Generator[torch.Tensor]:
        """
        Returns a generator where each iteration would optimize once.

        Yields:
            The loss per iteration.
        """

        for batch in self.dataloader:
            loss = self.train_step(batch)
            yield loss

    def train_step(self, batch: VectorPair) -> torch.Tensor:
        """
        Train one single step.
        """

        loss = self.compute_loss(batch)
        assert isinstance(loss, torch.Tensor)
        optimize_clip(self.optimizer, loss, self.compute_loss, self.max_grad_norm)
        return loss


class TrainLoss(typing.Protocol):
    """
    The training step function.
    """

    def __call__(self, batch: VectorPair) -> torch.Tensor: ...


class _LossFunc(typing.Protocol):
    "A loss function must have `(input, target) -> loss` signature."

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor: ...


class ComputeSlLoss(nn.Module):
    """
    The module that wraps a supervised learning
    """

    def __init__(self, module: nn.Module, loss_func: _LossFunc) -> None:
        super().__init__()
        self.module = module
        self.loss_func = loss_func

    def forward(self, batch: VectorPair) -> torch.Tensor:
        pred = self.module(batch.x)
        return self.loss_func(pred, batch.y)


@ctxl.contextmanager
def optimize(
    opt: optim.Optimizer, loss: torch.Tensor
) -> cabc.Generator[optim.Optimizer]:
    """
    The optimizer context.

    First `opt.zero_grad()`, `loss.backward()` then `opt.step()`.

    This allows you to enclose loss creation / modification in a scope.

    Args:
        opt: An `optim.Optimzier`.
        loss: the `torch.Tensor` that is a scalar.

    Note:
        Unlike normal context managers,
        the closing condition (`.step`) only executes
        if the scope does not raise an error.
    """

    opt.zero_grad()
    loss.backward()
    yield opt
    opt.step()


def optimize_clip(
    opt: optim.Optimizer,
    loss: torch.Tensor,
    module: nn.Module,
    max_grad_norm: float | None = None,
) -> None:
    """
    Optimize the loss.
    """

    with optimize(opt, loss):
        if max_grad_norm is not None:
            clip_gradients(module.parameters(), max_grad_norm)
