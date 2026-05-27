# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
from torch import nn

from aioway._utils import render_fcall
from aioway.hop import NnInit, find_nn_init
from aioway.modes import NnInitFn


class _ModuleOpts(typing.NamedTuple):
    module: type[nn.Module]
    options: dict[str, typing.Any]


def _layers():
    yield nn.Identity, {}

    yield nn.Linear, {"in_features": 3, "out_features": 5}
    yield nn.Bilinear, {"in1_features": 3, "in2_features": 3, "out_features": 5}

    yield nn.Conv1d, {"in_channels": 3, "out_channels": 4, "kernel_size": 3}
    yield nn.Conv2d, {"in_channels": 3, "out_channels": 4, "kernel_size": 3}
    yield nn.Conv3d, {"in_channels": 3, "out_channels": 4, "kernel_size": 3}

    yield nn.MaxPool1d, {"kernel_size": 3}
    yield nn.MaxPool2d, {"kernel_size": 3}
    yield nn.MaxPool3d, {"kernel_size": 3}

    yield nn.AvgPool1d, {"kernel_size": 3}
    yield nn.AvgPool2d, {"kernel_size": 3}
    yield nn.AvgPool3d, {"kernel_size": 3}

    yield nn.Embedding, {"num_embeddings": 10, "embedding_dim": 3}
    yield nn.EmbeddingBag, {"num_embeddings": 10, "embedding_dim": 3}

    yield nn.Dropout, {}
    yield nn.Dropout1d, {}
    yield nn.Dropout2d, {}
    yield nn.Dropout3d, {}

    yield nn.BatchNorm1d, {"num_features": 13}
    yield nn.BatchNorm2d, {"num_features": 13}
    yield nn.BatchNorm3d, {"num_features": 13}

    yield nn.InstanceNorm1d, {"num_features": 13}
    yield nn.InstanceNorm2d, {"num_features": 13}
    yield nn.InstanceNorm3d, {"num_features": 13}


def _layers_opts():
    for case in _layers():
        yield _ModuleOpts(*case)


@pytest.fixture(
    params=_layers_opts(),
    ids=lambda x: render_fcall(x.module.__name__, **x.options),
)
def layer_opts(request: pytest.FixtureRequest) -> _ModuleOpts:
    return request.param


def _losses():
    yield nn.L1Loss, {}
    yield nn.MSELoss, {}
    yield nn.CrossEntropyLoss, {}
    yield nn.CTCLoss, {}
    yield nn.NLLLoss, {}
    yield nn.KLDivLoss, {}
    yield nn.BCELoss, {}
    yield nn.BCEWithLogitsLoss, {}
    yield nn.SmoothL1Loss, {}


def _losses_opts():
    for case in _losses():
        yield _ModuleOpts(*case)


@pytest.fixture(
    params=_losses_opts(),
    ids=lambda x: render_fcall(x.module.__name__, **x.options),
)
def loss_opts(request: pytest.FixtureRequest) -> _ModuleOpts:
    return request.param


@pytest.fixture
def layer_thunk(layer_opts: _ModuleOpts):
    "The `NnInitFn` that are layers in `nn.Module`."
    cls, opts = layer_opts
    return NnInitFn(func=cls, args=(), kwargs=opts)


@pytest.fixture
def layer_nn_init(layer_thunk: NnInitFn) -> NnInit:
    "The `NnInit` from layers in `nn.Module`."
    result = find_nn_init(layer_thunk)
    assert result
    return result


@pytest.fixture
def loss_thunk(loss_opts: _ModuleOpts):
    cls, opts = loss_opts
    return NnInitFn(func=cls, args=(), kwargs=opts)


@pytest.fixture
def loss_nn_init(loss_thunk: NnInitFn) -> NnInit:
    "The `NnInit` from losses in `nn.Module`."
    result = find_nn_init(loss_thunk)
    assert result
    return result


@pytest.fixture
def opts(module_opts: _ModuleOpts):
    _, opts = module_opts
    return opts
