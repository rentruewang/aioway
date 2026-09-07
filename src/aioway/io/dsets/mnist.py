# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib
import typing

import tensordict as td
import torch
from torchrl.data import tensor_specs as tspecs
from torchvision import datasets
from torchvision import transforms as T

from .comps import InputTarget, InputTargetLikeDset

__all__ = ["MnistDataset"]


class MnistDataset(InputTargetLikeDset):
    def __init__(self) -> None:
        self._mnist = datasets.MNIST(pathlib.Path.home(), download=True)

    def __len__(self) -> int:
        return len(self._mnist)

    def __getitem__(self, idx) -> InputTarget:
        x, y = self._mnist[idx]
        x = T.ToTensor()(x)
        return InputTarget(x, y)

    @typing.no_type_check
    def __getitems__(self, idx):
        return td.stack([self[i] for i in idx])

    @property
    def input_tspec(self) -> tspecs.Unbounded:
        return tspecs.Unbounded(shape=torch.Size([28, 28]))

    @property
    def target_tspec(self) -> tspecs.Bounded:
        return tspecs.Bounded(low=0, high=9, shape=torch.Size([]), dtype=torch.long)

    @property
    def collate_fn(self):
        return lambda x: x
