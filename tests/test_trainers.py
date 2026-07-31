# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls

import pytest
import torch
from torch import nn

from aioway.emits import MlpEmitter
from aioway.io import TensorFrame
from aioway.spaces import Shape, ShapeSpace
from aioway.trainers import Trainer


@dcls.dataclass(frozen=True)
class _TensorFrame(TensorFrame):
    tensor: torch.Tensor

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx]

    def __getitems__(self, idx):
        return list(self.tensor[idx])

    @property
    def space(self):
        return ShapeSpace(Shape.parse(self.tensor.shape[1:]))


@pytest.fixture
def trainer():
    return Trainer(nn.MSELoss())


@pytest.fixture
def x():
    return _TensorFrame(torch.randn(100, 13))


@pytest.fixture
def y():
    return _TensorFrame(torch.randn(100, 11))


@pytest.fixture(autouse=True)
def mlp_emitter():
    with MlpEmitter([17, 19]).consider():
        yield


def test_trainer_fit(trainer, x, y):
    list(trainer.fit(x, y))


def test_trainer_fit_predict(trainer, x, y):
    list(trainer.fit(x, y))
    list(trainer.predict(x))
