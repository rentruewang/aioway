# Copyright (c) AIoWay Authors - All Rights Reserved

from collections import abc as cabc

from torch import nn, optim

from .static import BatchIter

__all__ = ["distil"]


def distil(
    batches: BatchIter,
    source: nn.Module,
    target: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
) -> cabc.Generator[None]:
    """
    Knowledge distilation of weights.
    """

    for batch in batches:
        loss = loss_fn(source(batch), target(batch))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        yield
