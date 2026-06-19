# Copyright (c) AIoWay Authors - All Rights Reserved

from collections import abc as cabc

from torch import optim

from aioway._comps import Iter, TensorIter, iter_dcls

__all__ = ["OptimizerIter"]


@iter_dcls
class OptimizerIter(Iter[None]):
    """
    The optimizer in the `Iter` format.
    """

    loss: TensorIter
    "The loss that would be optimized."

    optimizer: optim.Optimizer
    "The optimizer to call `.step()` on."

    def __post_init__(self) -> None:
        if self.loss.ndim != 0:
            raise ValueError(f"The input loss should be 0D, got {self.loss.shape=}.")

    def iterate(self) -> cabc.Generator[None]:
        for loss in self.loss:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            yield
