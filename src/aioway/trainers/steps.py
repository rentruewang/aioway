# Copyright (c) AIoWay Authors - All Rights Reserved

import abc

from torch import nn
import tensordict as td
import torch, typing


__all__ = ["TrainStep"]


class TrainStep(abc.ABC):
    "The training step function."

    BATCH_CLS: type = object

    def __call__(self, batch: typing.Any) -> torch.Tensor:
        assert isinstance(batch, self.BATCH_CLS)
        return self.forward(batch)

    @abc.abstractmethod
    def forward(self, batch: typing.Any) -> torch.Tensor:
        raise NotImplementedError
