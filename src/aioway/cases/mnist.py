# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib
import typing

import torch
from rich import progress
from torch import nn, optim
from torch.utils import data as dutils
from torchrl import data as rldata
from torchvision import datasets
from torchvision import transforms as T

from aioway.emits import ClfLogitHead, linear_regression
from aioway.io import Frame
from aioway.trainers import static_infer_step, static_train_step

__all__ = ["mnist", "train_test_split"]


class MnistDataset(dutils.Dataset):
    def __init__(self) -> None:
        self._mnist = datasets.MNIST(pathlib.Path.home(), download=True)

    def __len__(self) -> int:
        return len(self._mnist)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self._mnist[index]
        x = T.ToTensor()(x).flatten()
        return x, y

    @property
    def observ_spec(self) -> rldata.Unbounded:
        return rldata.Unbounded(shape=torch.Size([784]))

    @property
    def action_spec(self) -> rldata.Bounded:
        return rldata.Bounded(low=0, high=9, shape=torch.Size([]), dtype=torch.long)


def mnist() -> MnistDataset:
    return MnistDataset()


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

    module = ClfLogitHead(linear_regression)(dset.observ_spec, dset.action_spec)
    optimizer = optim.AdamW(module.parameters())
    loss_func = nn.CrossEntropyLoss()

    for x, y in progress.track(train_loader):
        pred, loss = static_train_step(module, optimizer, loss_func, x, y)
        print(loss)

    for x, y in progress.track(test_loader):
        _, loss = static_infer_step(module, loss_func, x, y)
        print(loss)


if __name__ == "__main__":
    main(64)
