# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn

from aioway.tasks import (
    DeepLiftExpl,
    Expl,
    FeatureAblationExpl,
    IntegratedGradientsExpl,
    KernelShapExpl,
    SaliencyExpl,
)


@pytest.fixture
def linear():
    return nn.Linear(4, 1)


@pytest.fixture
def input():
    return torch.randn(32, 4)


def _expls():
    yield SaliencyExpl()
    yield IntegratedGradientsExpl()
    yield DeepLiftExpl()
    yield FeatureAblationExpl()
    yield KernelShapExpl()


@pytest.fixture(params=_expls())
def expl(request):
    return request.param


def test_expl_on_module(linear: nn.Module, input: torch.Tensor, expl: Expl):
    attribution = expl(linear, input)
    assert all(isinstance(t, torch.Tensor) for t in attribution)
