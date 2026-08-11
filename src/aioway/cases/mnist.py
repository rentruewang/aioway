# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib
import typing

import tensordict as td
import torch
from torch import nn
from torch.utils import data as dutils
from torchrl import data as rldata
from torchvision import datasets

from aioway.io import Frame
from aioway.trainers import SupervisedTrainer, VectorPair

__all__ = ["mnist", "train_test_split"]


class MnistFrame(Frame):
    def __init__(self) -> None:
        self._mnist = datasets.MNIST(pathlib.Path.home())

    def __len__(self) -> int:
        return len(self._mnist)

    def __getitem__(self, index) -> VectorPair:
        x, y = self._mnist[index]
        x = x.flatten()
        return VectorPair(x, y)

    def __getitems__(self, index: list[int]) -> VectorPair:
        pairs = [self[i] for i in index]
        return td.stack(pairs, dim=0)

    @property
    def observ_spec(self) -> rldata.Unbounded:
        return rldata.Unbounded(shape=torch.Size([784]))

    @property
    def action_spec(self) -> rldata.Bounded:
        return rldata.Bounded(low=0, high=9, dtype=torch.long)


def mnist() -> MnistFrame:
    return MnistFrame()


@typing.no_type_check
def train_test_split[D: Frame](dataset: D, test_ratio: float) -> tuple[D, D]:
    assert 0 <= test_ratio <= 1

    test_samples = round(len(dataset) * test_ratio)
    train_samples = len(dataset) - test_samples
    return dutils.random_split(dataset, [train_samples, test_samples])


def main(batch_size: int):

    dset = mnist()
    train_dset, test_dset = train_test_split(dset, 0.1)

    train_loader = dutils.DataLoader(train_dset, batch_size=batch_size)
    test_loader = dutils.DataLoader(test_dset, batch_size=batch_size)

    # trainer = SupervisedTrainer(
    #     nn.MSELoss(),
    # )
