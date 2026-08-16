# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib
import typing

import lightning as L
import torch
from torch import nn, optim
from torch.utils import data as dutils
from torchrl import data as rldata
from torchvision import datasets
from torchvision import transforms as T

from aioway.nets import ClfLogitHead, linear_regression
from aioway.trainers import (
    StaticTrainer,
    TrainCfg,
)

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
def train_test_split[D: dutils.Dataset](dataset: D, test_ratio: float) -> tuple[D, D]:
    assert 0 <= test_ratio <= 1

    test_samples = round(len(dataset) * test_ratio)
    train_samples = len(dataset) - test_samples
    return dutils.random_split(dataset, [train_samples, test_samples])


def main(batch_size: int):
    fabric = L.Fabric()

    dset = mnist()
    train_dset, test_dset = train_test_split(dset, 0.1)

    module = ClfLogitHead(linear_regression)(dset.observ_spec, dset.action_spec)
    optimizer = optim.AdamW(module.parameters())
    loss_func = nn.CrossEntropyLoss()

    cfg = TrainCfg(batch_size=64, fabric=fabric)
    trainer = StaticTrainer(cfg, module, optimizer, loss_func)

    for pred in trainer.train_epoch(train_dset):
        print(pred.loss)
    for pred in trainer.infer_epoch(test_dset):
        print(pred.loss)


if __name__ == "__main__":
    main(64)
