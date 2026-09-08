# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn

from aioway.nn import NnArgs


@pytest.fixture
def linear():
    return nn.Linear(3, 5)


def _args():
    yield NnArgs({"input": torch.randn(7, 3)})
    yield NnArgs(torch.randn(7, 3))


@pytest.fixture(params=_args())
def args(request: pytest.FixtureRequest):
    return request.param


def test_args_apply(linear: nn.Linear, args: NnArgs):
    output = args.invoke(linear)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (7, 5)
