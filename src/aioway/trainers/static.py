# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc
from torch.utils import data as dutils
import torch
from torch import nn, optim
import lightning as L
from torch.nn import utils as nn_utils
import dataclasses as dcls

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


@dcls.dataclass(frozen=True)
class TrainCfg:
    """
    The config that differs each run by run.
    """

    batch_size: int
    "The batch size to use in training."

    max_grad_norm: float = 1
    "The maximum gradient norm. Defaults to 1"

    def __post_init__(self) -> None:
        if self.max_grad_norm < 0:
            raise ValueError(f"{self.max_grad_norm=} should be positive.")


@dcls.dataclass(frozen=True)
class StaticTrainer:
    """
    The trainer for typical training workflow.
    """

    module: nn.Module
    "The module to train."

    loss_func: LossFunc
    "The loss function to compute the losses."

    optimizer: optim.Optimizer
    "The optimizer to use."

    fabric: L.Fabric
    "The fabric instance."

    def train_epoch(
        self, cfg: TrainCfg, train_dataset: dutils.Dataset
    ) -> cabc.Generator[None]:
        for x, y in dutils.DataLoader(train_dataset, batch_size=cfg.batch_size):
            self.train_step(cfg, x, y)
            yield

    def train_step(
        self, cfg: TrainCfg, x: torch.Tensor, y: torch.Tensor
    ) -> PredLossPair:
        "Train a module in a static (non interactive) manner."

        inferred = self.infer_step(x, y)

        self.optimizer.zero_grad()
        inferred.loss.backward()

        self.clip_gradients(cfg)

        self.optimizer.step()
        return inferred

    def infer_step(self, x: torch.Tensor, y: torch.Tensor) -> PredLossPair:
        pred = self.module(x)
        loss = self.loss_func(pred, y)
        return PredLossPair(pred, loss)

    def clip_gradients(self, cfg: TrainCfg) -> None:
        self.fabric.clip_gradients(
            self.module, self.optimizer, max_norm=cfg.max_grad_norm
        )
