# Copyright (c) AIoWay Authors - All Rights Reserved


import pytest
from torch import nn

from aioway.torch.nn import NnInitThunk, NnLayerUFunc, NnLossUFunc, NnUFunc, nn_ufunc


@pytest.fixture
def layer_nn_init(layer_thunk: NnInitThunk) -> NnLayerUFunc:
    "The `NnInit` from layers in `nn.Module`."
    result = nn_ufunc(layer_thunk.func, *layer_thunk.args, **layer_thunk.kwargs)
    assert isinstance(result, NnLayerUFunc)
    return result


@pytest.fixture
def loss_nn_init(loss_thunk: NnInitThunk) -> NnLossUFunc:
    "The `NnInit` from losses in `nn.Module`."
    result = nn_ufunc(loss_thunk.func, *loss_thunk.args, **loss_thunk.kwargs)
    assert isinstance(result, NnLossUFunc)
    return result


def test_layer_nn_init(layer_nn_init: NnUFunc):
    assert isinstance(layer_nn_init, NnUFunc)


def test_layer_nn_init_module(layer_nn_init: NnUFunc):
    module = layer_nn_init.module
    assert isinstance(module, nn.Module)


def test_loss_nn_init(loss_nn_init: NnUFunc):
    assert isinstance(loss_nn_init, NnUFunc)


def test_loss_nn_init_module(loss_nn_init: NnUFunc):
    module = loss_nn_init.module
    assert isinstance(module, nn.Module)


def test_sequential():
    seq_init: NnUFunc = nn_ufunc(
        nn.Sequential,
        nn.Linear(1, 2),
        nn.Linear(2, 3),
        nn.Linear(3, 4),
    )
    assert seq_init
    assert seq_init._func == nn.Sequential
    assert isinstance(seq_init.module, nn.Sequential)
