# Copyright (c) AIoWay Authors - All Rights Reserved


from torch import nn

from aioway.hop import NnInit, Sequential, find_nn_init
from aioway.modes import NnInitFn


def test_layer_nn_init(layer_nn_init: NnInit):
    assert isinstance(layer_nn_init, NnInit)


def test_layer_nn_init_module(layer_nn_init: NnInit):
    module = layer_nn_init()
    assert isinstance(module, nn.Module)


def test_loss_nn_init(loss_nn_init: NnInit):
    assert isinstance(loss_nn_init, NnInit)


def test_loss_nn_init_module(loss_nn_init: NnInit):
    module = loss_nn_init()
    assert isinstance(module, nn.Module)


def test_sequential():
    seq_init = find_nn_init(
        NnInitFn(
            func=nn.Sequential,
            args=(nn.Linear(1, 2), nn.Linear(2, 3), nn.Linear(3, 4)),
            kwargs={},
        )
    )
    assert seq_init
    assert isinstance(seq_init, Sequential)
    assert isinstance(seq_init(), nn.Sequential)
    assert len(seq_init.modules) == 3
