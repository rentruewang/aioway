# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
from torch import nn

from aioway._common import render_fcall
from aioway.might import Might, find_might


class _ModuleOpts(typing.NamedTuple):
    module: type[nn.Module]
    options: dict[str, typing.Any]


def _cases():
    yield _ModuleOpts(
        nn.Linear,
        {"in_features": 3, "out_features": 5},
    )

    yield _ModuleOpts(
        nn.Bilinear,
        {"in1_features": 3, "in2_features": 3, "out_features": 5},
    )

    yield _ModuleOpts(
        nn.Conv1d,
        {"in_channels": 3, "out_channels": 4, "kernel_size": 3},
    )

    yield _ModuleOpts(
        nn.Conv2d,
        {"in_channels": 3, "out_channels": 4, "kernel_size": 3},
    )

    yield _ModuleOpts(
        nn.Conv3d,
        {"in_channels": 3, "out_channels": 4, "kernel_size": 3},
    )


@pytest.fixture(params=_cases(), ids=lambda x: render_fcall(x[0].__name__, **x[1]))
def might(request: pytest.FixtureRequest):
    cls, kwargs = request.param
    return find_might(cls, **kwargs)


def test_might_init(might: Might):
    assert isinstance(might, Might)
    assert repr(might).startswith("might::")


def test_might_do(might: Might):
    module = might.do()
    assert isinstance(module, nn.Module)
