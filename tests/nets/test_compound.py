# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn

from aioway.emits import BuiltModule, CompoundBuilder


@pytest.fixture
def built_mlp():
    builder = CompoundBuilder()
    input = builder.input("input")
    hidden_1 = builder.thunk(nn.Linear(in_features=5, out_features=10), input)
    hidden_2 = builder.thunk(nn.Linear(in_features=10, out_features=10), hidden_1)

    module = builder.output(hidden_2)
    return module


def test_mlp(built_mlp: BuiltModule):
    t = torch.randn(13, 5)
    out = built_mlp(input=t)
    assert out.shape == (13, 10)


def test_loss():
    builder = CompoundBuilder()
    input = builder.input("input")
    hidden_1 = builder.thunk(nn.Linear(in_features=5, out_features=10), input)
    hidden_2 = builder.thunk(nn.Linear(in_features=10, out_features=5), hidden_1)
    loss = builder.thunk(nn.MSELoss(), hidden_2, input)

    graph = builder.output(loss)

    t = torch.randn(13, 5)

    out = graph(input=t)
    assert out.shape == ()
