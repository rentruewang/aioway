# Copyright (c) AIoWay Authors - All Rights Reserved

import torch

from aioway._ufuncs import CompoundBuilder
from aioway.torch.nn import Linear, MSELoss


def test_mlp():
    builder = CompoundBuilder()
    input = builder.input("input")
    hidden_1 = builder.thunk(Linear(in_features=5, out_features=10).ufunc, input)
    hidden_2 = builder.thunk(Linear(in_features=10, out_features=10).ufunc, hidden_1)

    module = builder.output(hidden_2)

    t = torch.randn(13, 5)
    out = module(input=t)
    assert out.shape == (13, 10)


def test_loss():
    builder = CompoundBuilder()
    input = builder.input("input")
    hidden_1 = builder.thunk(Linear(in_features=5, out_features=10).ufunc, input)
    hidden_2 = builder.thunk(Linear(in_features=10, out_features=5).ufunc, hidden_1)
    loss = builder.thunk(MSELoss().ufunc, hidden_2, input)

    graph = builder.output(loss)
    t = torch.randn(13, 5)
    out = graph(input=t)
    assert out.shape == ()
