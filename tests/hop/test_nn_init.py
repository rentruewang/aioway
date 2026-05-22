# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
from torch import nn

from aioway._utils import render_fcall
from aioway.modes import NnInitFn
from aioway.op import NnInit, Sequential, find_nn_init


class _ModuleOpts(typing.NamedTuple):
    module: type[nn.Module]
    options: dict[str, typing.Any]


def _cases():
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

    yield nn.L1Loss, {}
    yield nn.MSELoss, {}
    yield nn.CrossEntropyLoss, {}
    yield nn.CTCLoss, {}
    yield nn.NLLLoss, {}
    yield nn.KLDivLoss, {}
    yield nn.BCELoss, {}
    yield nn.BCEWithLogitsLoss, {}
    yield nn.SmoothL1Loss, {}

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


def _module_opts():
    for case in _cases():
        yield _ModuleOpts(*case)


@pytest.fixture(
    params=_module_opts(),
    ids=lambda x: render_fcall(x.module.__name__, **x.options),
)
def module_opts(request: pytest.FixtureRequest) -> _ModuleOpts:
    return request.param


@pytest.fixture
def nn_init(module_opts: _ModuleOpts):
    cls, opts = module_opts
    result = find_nn_init(NnInitFn(func=cls, args=(), kwargs=opts))
    assert result
    return result


@pytest.fixture
def opts(module_opts: _ModuleOpts):
    _, opts = module_opts
    return opts


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
