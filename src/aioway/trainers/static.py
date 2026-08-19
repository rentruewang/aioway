# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway.spaces import Space, SpaceCompat
import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

import lightning as L
import torch
from rich import progress
from torch import nn, optim
from torch.utils import data as dutils
from torchrl.data import tensor_specs as tspecs
import tensordict as td
from aioway.dsets import Dset, InputTargetLikeDset, InputTarget

__all__ = ["LossFunc", "PredLossPair", "StaticTrainer", "TrainCfg"]


class LossFunc(typing.Protocol):
    """
    A loss function should have the format (input, target) -> loss.
    """

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor: ...


class PredLossPair(typing.NamedTuple):
    "The prediction and loss pair."

    pred: torch.Tensor
    loss: torch.Tensor


class ObservActionSpace(abc.ABC):
    @property
    @abc.abstractmethod
    def observ_space(self) -> tspecs.TensorSpec:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_space(self) -> tspecs.TensorSpec:
        raise NotImplementedError


@dcls.dataclass(frozen=True, kw_only=True)
class TrainCfg:
    """
    The config that differs each run by run.
    """

    batch_size: int
    "The batch size to use in training."

    fabric: L.Fabric
    "The fabric instance."

    shuffle: bool = True
    "Whether to shuffle or not. Defaults to `True`."

    max_grad_norm: float = 1
    "The maximum gradient norm. Defaults to 1."

    progress_bar: bool = True
    "Whether or not to use a progress bar."

    def __post_init__(self) -> None:
        if self.max_grad_norm < 0:
            raise ValueError(f"{self.max_grad_norm=} should be positive.")

    def make_data_loader(self, dataset: Dset) -> dutils.DataLoader:
        loader = dutils.DataLoader(
            dataset.__dataset__(),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=dataset.__collate_fn__,
        )
        return typing.cast(dutils.DataLoader, self.fabric.setup_dataloaders(loader))


class TerminateTraining(StopIteration):
    """
    Signal to stop the current training loop.
    """

    value: PredLossPair
    "The value should still be a prediction and loss pair."


class TrainingContext(typing.Protocol):
    def __call__(self, trainer: StaticTrainer, state: LoopState):
        pass


@dcls.dataclass(frozen=True)
class LoopState:
    batch_idx: int
    "The batch index of the current batch."

    total_size: int | None = None
    "The total size. For stream this would be `None`."


class BatchIter[T: td.TensorClass | td.TensorDict = typing.Any](typing.Protocol):
    """
    The constraints that trainer uses to define what trainer handles.
    """

    def __iter__(self) -> cabc.Iterator[T]:
        "Iterates and yield batches."

        ...

    def __space__(self) -> Space[T]:
        """
        The space constraining the output of `__iter__`.
        """


@dcls.dataclass(frozen=True)
class IterableBatchIter:
    "Wraps an iterable and space."

    iterable: cabc.Iterable
    space: SpaceCompat

    def __iter__(self):
        yield from self.iterable

    def __space__(self) -> SpaceCompat:
        return self.space


class StaticTrainer:
    """
    The trainer for typical training workflow.
    """

    def __init__(
        self,
        cfg: TrainCfg,
        module: nn.Module,
        optimizer: optim.Optimizer,
        loss_func: LossFunc,
    ):
        self._cfg = cfg

        self._module, self._optimizer = self.fabric.setup(module, optimizer)

        assert isinstance(self._module, nn.Module)
        assert isinstance(self._optimizer, optim.Optimizer)

        self._loss_func = loss_func

    def train_dataset_epoch(
        self, dataset: InputTargetLikeDset
    ) -> cabc.Generator[PredLossPair]:
        iterable = self._data_loader(dataset)
        yield from self.train_epoch(iterable)

    def train_epoch(self, batch_iter: BatchIter[InputTarget]):
        for pair in batch_iter:
            x, y = pair.input, pair.target

            try:
                inferred = self.train_step(x, y)
            except TerminateTraining as tt:
                yield tt.value
                return
            else:
                yield inferred

    def validate_dataset_epoch(
        self, dataset: InputTargetLikeDset
    ) -> cabc.Generator[PredLossPair]:
        iterable = self._data_loader(dataset)
        yield from self.validate_epoch(iterable)

    def validate_epoch(self, batch_iter: BatchIter[InputTarget]):
        for pair in batch_iter:
            x, y = pair.input, pair.target
            inferred = self.infer_step(x, y)
            yield inferred

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> PredLossPair:
        "Train a module in a static (non interactive) manner."

        inferred = self.infer_step(x, y)

        self.optimizer.zero_grad()
        self.backward(inferred.loss)
        self.clip_gradients()
        self.optimizer.step()

        return inferred

    def infer_step(self, x: torch.Tensor, y: torch.Tensor) -> PredLossPair:
        pred = self.module(x)
        loss = self.loss_func(pred, y)
        return PredLossPair(pred, loss)

    def backward(self, loss: torch.Tensor) -> None:
        self.fabric.backward(loss)

    def clip_gradients(self) -> None:
        self.fabric.clip_gradients(
            self.module,
            self.optimizer,
            max_norm=self.cfg.max_grad_norm,
        )

    @property
    def fabric(self) -> L.Fabric:
        return self.cfg.fabric

    @property
    def cfg(self) -> TrainCfg:
        "The configuration used in training."
        return self._cfg

    @property
    def module(self) -> nn.Module:
        "The module to train."
        return self._module

    @property
    def loss_func(self) -> LossFunc:
        "The loss function to compute the losses."
        return self._loss_func

    @property
    def optimizer(self) -> optim.Optimizer:
        "The optimizer to use."
        return self._optimizer

    def _data_loader(self, dataset: InputTargetLikeDset):
        loader: cabc.Iterable = self.cfg.make_data_loader(dataset)

        if self.cfg.progress_bar:
            loader = progress.track(loader)

        return IterableBatchIter(loader, dataset.__space__())
