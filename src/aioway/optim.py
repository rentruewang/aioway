# Copyright (c) AIoWay Authors - All Rights Reserved

from collections import abc as cabc

from torch import optim

from aioway.hop import Hop, TensorHop, hop_dcls

__all__ = ["OptimizerHop"]


@hop_dcls
class OptimizerHop(Hop[None]):
    """
    The optimizer in the `Hop` format.
    """

    loss: TensorHop
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
