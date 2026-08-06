# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn, optim
from torch.utils import data

from aioway.trainers import SupervisedLoss, SupervisedTrainer, VectorPair


class _RandomDataset(data.Dataset):
    def __init__(self, size: int, in_feats: int, out_feats: int) -> None:
        super().__init__()
        self._x = torch.randn(size, in_feats)
        self._y = torch.randn(size, out_feats)

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, index):
        return VectorPair(self._x[index], self._y[index])

    __getitems__ = __getitem__


@pytest.fixture(params=[3, 5, 7])
def in_feats(request):
    return request.param


@pytest.fixture(params=[3, 5, 7])
def out_feats(request):
    return request.param


@pytest.fixture(params=[3, 5, 7])
def size(request):
    return request.param


@pytest.fixture
def dataset(size, in_feats, out_feats):
    return _RandomDataset(size, in_feats, out_feats)


@pytest.fixture
def data_loader(dataset):
    return data.DataLoader(dataset, batch_size=2, collate_fn=lambda x: x)


@pytest.fixture
def module(in_feats, out_feats):
    return nn.Linear(in_feats, out_feats)


def _train_loss():
    yield nn.MSELoss()
    yield nn.SmoothL1Loss()


@pytest.fixture(params=_train_loss())
def train_loss(request: pytest.FixtureRequest, module: nn.Module):
    return SupervisedLoss(module, request.param)


@pytest.fixture
def opt(module: nn.Module):
    return optim.Adam(module.parameters())


@pytest.fixture
def trainer(
    train_loss: nn.Module,
    opt: optim.Optimizer,
    data_loader: data.DataLoader,
):
    return SupervisedTrainer(
        train_loss,
        optimizer=opt,
        dataloader=data_loader,
        max_grad_norm=1,
    )


def test_train_epoch(trainer: SupervisedTrainer) -> None:
    for loss in trainer.train_epoch():
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
