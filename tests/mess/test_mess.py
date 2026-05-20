# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
from torch import nn

from aioway.mess import Mess, MessInit
from aioway.mess.mess import find_mess
from aioway.renders import render_fcall


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
def mess(module_opts: _ModuleOpts) -> Mess:
    cls, _ = module_opts
    return find_mess(cls)


@pytest.fixture
def opts(module_opts: _ModuleOpts):
    _, opts = module_opts
    return opts


def test_mess(mess: Mess):
    assert isinstance(mess, Mess)


def test_mess_init(mess: Mess):
    init = mess.init
    assert issubclass(init, MessInit)


def test_init_module(mess: Mess, opts: dict[str, typing.Any]):
    module = mess.module(**opts)
    assert isinstance(module, nn.Module)
    assert isinstance(module, mess.nn_type)


def test_sequential():
    seq = find_mess(nn.Sequential).module(
        nn.Linear(1, 2), nn.Linear(2, 3), nn.Linear(3, 4)
    )
    assert isinstance(seq, nn.Sequential)
    assert len(seq) == 3
