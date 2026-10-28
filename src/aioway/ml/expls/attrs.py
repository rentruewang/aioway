# Copyright (c) AIoWay Authors - All Rights Reserved

"The explainer based on `captum.attr`."

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from captum import attr
from torch import nn

from aioway.io import Dset
from aioway.ml.tasks import Task

__all__ = ["CaptumAttrTask"]


@dcls.dataclass
class CaptumAttrTask(Task, abc.ABC):
    """
    The explainer based on `captum.attr.Attribution`.
    """

    build_attr: cabc.Callable[[nn.Module], attr.Attribution]
    "The attr build function."

    batch_size: int = 256
    "The batch size to use to iterate over the dataset."

    def __call__(self, dataset: Dset, module: nn.Module) -> torch.Tensor:
        """
        This function corresponds to `captum.attr.*`'s `.attribute` method

        Args:
            dataset: Inputs that will be fed to the `module`.
            module: An `nn.Module`, must output a scalar tenosr.
        """

        attr = self.build_attr(module)
        total = 0.0

        n = 0
        for x, y in dataset.to_dataloader(batch_size=self.batch_size):
            attr = attr.attribute(x, target=y, **self.kwargs).detach()
            total += attr.abs().sum(0)
            n += len(x)

        assert isinstance(total, torch.Tensor)
        return total / n
