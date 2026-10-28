# Copyright (c) AIoWay Authors - All Rights Reserved

"The explainer based on `captum`."

import typing
from collections import abc as cabc

import torch
from captum import attr
from torch import nn
from torch.utils import data as du

from aioway.io import Dset
from aioway.ml.tasks import Task


class CaptumAttrTask(Task):
    def __init__(
        self, build: cabc.Callable[[nn.Module], attr.Attribution], batch_size: int = 256
    ) -> None:
        self._build = build
        self._batch_size = batch_size

    def __call__(self, dataset: Dset, module: nn.Module) -> torch.Tensor:
        """
        This function corresponds to `captum.attr.*`'s `.attribute` method

        Args:
            dataset: Inputs that will be fed to the `module`.
            module: An `nn.Module`, must output a scalar tenosr.
        """

        attr = self._build(module)
        total = 0.0

        n = 0
        for x, y in du.DataLoader(dataset.to_dataset(), batch_size=self._batch_size):
            attr = attr.attribute(x, target=y).detach()
            total += attr.abs().sum(0)
            n += len(x)

        assert isinstance(total, torch.Tensor)
        return total / n


class CaptumConceptTask(Task):
    def __call__(self, dataset: Dset, module: nn.Module) -> typing.Any:
        raise NotImplementedError


class CaptumInfluenceTask(Task):
    def __call__(self, dataset: Dset, module: nn.Module) -> typing.Any:
        raise NotImplementedError
