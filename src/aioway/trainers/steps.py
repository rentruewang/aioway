# Copyright (c) AIoWay Authors - All Rights Reserved

"The stepping thunks."

import abc
import dataclasses as dcls
import typing

import lightning as L
import torch
from torch import nn, optim

from aioway._thunks import Thunk
from aioway._utils import FloatArray

__all__ = ["Step", "ValidateStep", "TrainStep", "IncSkLearnStep"]


class Step[T](Thunk[T], abc.ABC):
    """
    The training step function. Invoking the `Step` updates the modules.

    Since the concept of stepping as computation is more or less the same,
    but not the data, data is stored in `Step` subclass's attributes,
    to maintain a consistent API across multiple different types of steps.
    """

    @abc.abstractmethod
    def __call__(self) -> T:
        """
        Perform the stepping.
        """

        raise NotImplementedError


class LossFunc[I, T](typing.Protocol):
    """
    A loss function should have the format (input, target) -> loss.
    """

    def __call__(self, input: I, target: T) -> torch.Tensor: ...


class PredLossPair[O](typing.NamedTuple):
    "The prediction and loss pair."

    pred: O
    "The prediction made by the module."

    loss: torch.Tensor
    "The loss (scalar tensor)."


@dcls.dataclass(frozen=True)
class ValidateStep[I, T](Step[PredLossPair]):
    module: nn.Module
    "The module to validate."

    loss_func: LossFunc
    "The loss function to compute the losses."

    input: I
    "The input data."

    target: T
    "The target label."

    @typing.override
    def __call__(self) -> PredLossPair:
        pred = self.module(self.input)
        loss = self.loss_func(pred, self.target)
        return PredLossPair(pred, loss)


@dcls.dataclass(frozen=True)
class TrainStep(Step[PredLossPair]):
    """
    A supervised training step has input and target.
    """

    validate: ValidateStep
    optimizer: optim.Optimizer
    "The optimizer with the parameters."

    fabric: L.Fabric
    "The fabric instance."

    max_norm: float | None = None
    "The maximum norm."

    def __call__(self) -> PredLossPair:
        "Train a module in a static (non interactive) manner."

        inferred = self.validate()

        self.optimizer.zero_grad()
        self.backward(inferred.loss)
        self.clip_gradients()
        self.optimizer.step()

        return inferred

    def backward(self, loss: torch.Tensor) -> None:
        self.fabric.backward(loss)

    def clip_gradients(self) -> None:
        if self.max_norm is None:
            return

        self.fabric.clip_gradients(self.module, self.optimizer, max_norm=self.max_norm)

    @property
    def module(self) -> nn.Module:
        "The module to train."

        return self.validate.module


class IncSkLearn(typing.Protocol):
    def partial_fit(self, x: FloatArray, /): ...

    def transform(self, x: FloatArray, /): ...


class IncSkLearnStep(Step[FloatArray]):
    algo: IncSkLearn
    "The algorithm that supports partial fitting."

    data: FloatArray
    "The data used in fitting."

    def __call__(self) -> FloatArray:
        self.algo.partial_fit(self.data)
        return self.algo.transform(self.data)
