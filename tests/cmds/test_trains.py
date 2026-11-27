# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
import torch
from torch import nn, optim

from aioway.cmds import Eval, Train
from aioway.cmds.nets import float_list
from aioway.io import InputTarget, InputTargetLikeDset


class _LinearDset(InputTargetLikeDset):
    "`y = x @ w`, something a linear layer can learn."

    def __init__(self) -> None:
        self._x = torch.randn(256, 4)
        self._y = self._x @ torch.randn(4, 1)

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, idx):
        return InputTarget(input=self._x[idx], target=self._y[idx])

    @typing.no_type_check
    def __getitems__(self, idx):
        return InputTarget(input=self._x[idx], target=self._y[idx])

    @property
    def input_domain(self):
        return float_list(4)

    @property
    def target_domain(self):
        return float_list(1)

    @property
    def collate_fn(self):
        return lambda x: x


def _train_cases():
    yield Train(batch_size=32)
    yield Train(batch_size=32, max_grad_norm=1)
    yield Train(batch_size=32, progress_bar=True)


@pytest.fixture(params=_train_cases())
def train(request) -> Train:
    return request.param


def test_loss_goes_down(train: Train):
    dset = _LinearDset()
    model = nn.Linear(4, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss = nn.MSELoss()
    evaluate = Eval(batch_size=32)

    before = evaluate(model, loss, dset)
    for _ in range(5):
        train(model, optimizer, loss, dset)
    after = evaluate(model, loss, dset)

    assert after < before
