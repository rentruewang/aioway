# Copyright (c) AIoWay Authors - All Rights Reserved

import torch

from aioway._ufuncs import InputBuilderNode, ThunkBuilderNode
from aioway.torch.nn import Linear, MSELoss


def test_mlp():
    input = InputBuilderNode("input")
    hidden_1 = ThunkBuilderNode(Linear(in_features=5, out_features=10).ufunc, input)
    hidden_2 = ThunkBuilderNode(Linear(in_features=10, out_features=10).ufunc, hidden_1)

    module = hidden_2.build()

    t = torch.randn(13, 5)
    out = module(input=t)
    assert out.shape == (13, 10)


def test_loss():
    input = InputBuilderNode("input")
    hidden_1 = ThunkBuilderNode(Linear(in_features=5, out_features=10).ufunc, input)
    hidden_2 = ThunkBuilderNode(Linear(in_features=10, out_features=5).ufunc, hidden_1)
    loss = ThunkBuilderNode(MSELoss().ufunc, hidden_2, input)

    graph = loss.build()
    t = torch.randn(13, 5)
    out = graph(input=t)
    assert out.shape == ()
