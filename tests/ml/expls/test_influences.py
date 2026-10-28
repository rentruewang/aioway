# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn
from torch.utils import data as du

from aioway.ml import (
    BuildNaiveInfluenceFunction,
    BuildTracInCP,
    BuildTracInCPFast,
    CaptumInfluenceTask,
)


@pytest.fixture
def linear():
    return nn.Sequential(nn.Linear(4, 3))


@pytest.fixture
def corpus():
    return du.TensorDataset(torch.randn(16, 4), torch.randint(0, 3, (16,)))


@pytest.fixture
def dset():
    return du.TensorDataset(torch.randn(32, 4), torch.randint(0, 3, (32,)))


def _builds():
    yield BuildTracInCP()
    yield BuildTracInCPFast(final_fc_layer="0")
    yield BuildNaiveInfluenceFunction(layers=["0"])


@pytest.fixture(params=_builds())
def build(request):
    return request.param


def test_influence_on_module(
    linear: nn.Module,
    corpus: du.Dataset,
    dset: du.Dataset,
    build: CaptumInfluenceTask.Strategy,
):
    influence = CaptumInfluenceTask(build, corpus, batch_size=8)(dset, linear)
    assert influence.shape == (16,)
