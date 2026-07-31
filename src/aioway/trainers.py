# Copyright (c) AIoWay Authors - All Rights Reserved

import torch
from torch import nn, optim

from aioway.emits import emit_one
from aioway.io import Dset
from aioway.relalg import LoaderOpt
from aioway.spaces import StatelessSpace

__all__ = ["Trainer"]


class Trainer:
    def __init__(self, loss_func: nn.Module) -> None:
        self._loss_func = loss_func
        self._module: nn.Module | None = None

    def fit(self, x: Dset, y: Dset, loader_opt: LoaderOpt = LoaderOpt()):
        self._module = emit_one(StatelessSpace(x.__space__(), y.__space__()))
        assert isinstance(self.module, nn.Module), self.module
        optimizer = optim.AdamW(self.module.parameters())

        for batch_x, batch_y in zip(x(loader_opt), y(loader_opt)):
            out = self.module(batch_x)
            loss: torch.Tensor = self._loss_func(out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            yield loss

    def predict(self, x: Dset, loader_opt: LoaderOpt = LoaderOpt()):
        for batch_x in x(loader_opt):
            yield self.module(batch_x)

    @property
    def module(self) -> nn.Module:
        assert self._module is not None
        return self._module
