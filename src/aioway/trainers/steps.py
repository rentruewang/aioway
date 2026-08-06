# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
import abc
from collections import abc as cabc

import torch
from torch import nn
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
