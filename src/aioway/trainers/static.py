# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

import lightning as L
import torch
from rich import progress
from torch import nn, optim
from torch.utils import data as dutils
from torchrl import data as rldata

from aioway.dsets import Dset, InputTargetLikeDset

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
    def observ_spec(self) -> rldata.TensorSpec:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_spec(self) -> rldata.TensorSpec:
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

    def train_epoch(self, dataset: InputTargetLikeDset) -> cabc.Generator[PredLossPair]:
        loader = self.cfg.make_data_loader(dataset)
        for pair in progress.track(loader):
            x, y = pair.input, pair.target
            inferred = self.train_step(x, y)
            yield inferred

    def infer_epoch(self, dataset: InputTargetLikeDset) -> cabc.Generator[PredLossPair]:
        loader = self.cfg.make_data_loader(dataset)
        for pair in progress.track(loader):
            x, y = pair.input, pair.target
            inferred = self.infer_step(x, y)
            yield inferred

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> PredLossPair:
        "Train a module in a static (non interactive) manner."

        inferred = self.infer_step(x, y)

        self.optimizer.zero_grad()
        self.fabric.backward(inferred.loss)

        self.clip_gradients()

        self.optimizer.step()
        return inferred

    def infer_step(self, x: torch.Tensor, y: torch.Tensor) -> PredLossPair:
        pred = self.module(x)
        loss = self.loss_func(pred, y)
        return PredLossPair(pred, loss)

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
