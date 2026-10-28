# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn
from torch.utils import data as du

from aioway.ml import (
    BuildDeepLift,
    BuildFeatureAblation,
    BuildIntegratedGradients,
    BuildKernelShap,
    BuildSaliency,
    CaptumAttrTask,
)


@pytest.fixture
def linear():
    return nn.Linear(4, 1)


@pytest.fixture
def dset():
    return du.TensorDataset(torch.randn(32, 4), torch.zeros(32, dtype=torch.long))


def _builds():
    yield BuildSaliency()
    yield BuildIntegratedGradients()
    yield BuildDeepLift()
    yield BuildFeatureAblation()
    yield BuildKernelShap()


@pytest.fixture(params=_builds())
def build(request):
    return request.param


@pytest.fixture
def batch_size():
    return 8


def test_attr_on_module(
    linear: nn.Module,
    dset: du.Dataset,
    build: CaptumAttrTask.Strategy,
    batch_size: int,
):
    attribution = CaptumAttrTask(build, batch_size=batch_size)(dset, linear)
    assert attribution.shape == (4,)
