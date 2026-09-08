# Copyright (c) AIoWay Authors - All Rights Reserved

import torch
from aioway.nn import NnArgs
from torch import nn
import pytest


@pytest.fixture
def linear():
    return nn.Linear(3, 5)


@pytest.fixture
def args():
    return NnArgs({"input": torch.randn(7, 3)})


def test_args_apply(linear: nn.Linear, args: NnArgs):
    output = args.invoke(linear)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (7, 5)
