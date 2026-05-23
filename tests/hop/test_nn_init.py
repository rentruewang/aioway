# Copyright (c) AIoWay Authors - All Rights Reserved


from torch import nn

from aioway.hop import NnInit, Sequential, find_nn_init
from aioway.modes import NnInitFn


def test_nn_init(nn_init: NnInit):
    assert isinstance(nn_init, NnInit)


def test_nn_init_module(nn_init: NnInit):
    module = nn_init.do()
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
    assert isinstance(seq_init.do(), nn.Sequential)
    assert len(seq_init.modules) == 3
