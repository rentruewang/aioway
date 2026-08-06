# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import contextlib as ctxl
import typing
from collections import abc as cabc

import torch
from torch import nn, optim
from torch.nn import utils as nn_utils
from torch.utils import data

__all__ = ["TrainLoss"]


class TrainLoss[T](nn.Module, abc.ABC):
    """
    The training step function.

    It is an `nn.Module` s.t. we can directly use optimizers on it.
    """

    BATCH_CLS: typing.ClassVar[type] = object
    "The batching class to use."

    module: nn.Module
    "The module to optimize."

    def __call__(self, batch: T) -> torch.Tensor:
        assert isinstance(batch, self.BATCH_CLS)
        return self.forward(batch)

    @abc.abstractmethod
    def forward(self, batch: T) -> torch.Tensor:
        raise NotImplementedError

    def for_data_loader(
        self, loader: data.DataLoader[T]
    ) -> cabc.Generator[torch.Tensor]:
        """
        Compute the losses over an entire data loader.
        """

        for batch in loader:
            yield self(batch)


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
            nn_utils.clip_grad_norm_(module.parameters(), max_norm=max_grad_norm)
