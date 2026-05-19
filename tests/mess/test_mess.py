# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
from torch import nn

from aioway.mess import Mess, MessInit, Sequential
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
def mess_init(request: pytest.FixtureRequest) -> MessInit:
    cls, kwargs = request.param
    return Mess.find(cls).init(**kwargs)


def test_mess_init(mess_init: MessInit):
    assert isinstance(mess_init, MessInit)
    assert repr(mess_init).startswith("mess_init::")


def test_mess_init_do(mess_init: MessInit):
    module = mess_init.do()
    assert isinstance(module, nn.Module)


def test_sequential():
    seq = Mess.find(nn.Sequential).init(
        nn.Linear(1, 2), nn.Linear(2, 3), nn.Linear(3, 4)
    )
    assert isinstance(seq, MessInit)
    assert isinstance(seq, Sequential)
    assert len(seq.modules) == 3
