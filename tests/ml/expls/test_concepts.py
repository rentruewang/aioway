# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn
from torch.utils import data as du

from aioway.ml import BuildTCAV, CaptumConceptTask


class _Dataset(du.Dataset):
    def __init__(self, tensors: torch.Tensor | list[torch.Tensor]) -> None:
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, index):
        return self.tensors[index]


@pytest.fixture
def mlp():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))


@pytest.fixture
def concepts():
    return {
        "high": _Dataset(torch.randn(16, 4) + 1),
        "low": _Dataset(torch.randn(16, 4) - 1),
    }


@pytest.fixture
def random_set():
    return _Dataset(torch.randn(16, 4))


@pytest.fixture
def dset():
    return du.TensorDataset(torch.randn(32, 4), torch.randint(0, 3, (32,)))


def _builds():
    yield BuildTCAV(layers=["module.0", "module.2"])


@pytest.fixture(params=_builds())
def build(request):
    return request.param


def test_concept_on_module(
    mlp: nn.Module,
    concepts: dict[str, du.Dataset],
    random_set: du.Dataset,
    dset: du.Dataset,
    build: CaptumConceptTask.Strategy,
):
    task = CaptumConceptTask(build, concepts, random_set, batch_size=8)
    scores = task(dset, mlp)
    assert scores.shape == (2, 2)
