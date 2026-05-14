# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib
import random

import pytest
import torch
from numpy import random as npr
from rich import traceback

from aioway.fn import fake_fn, track_fn

from .fake import batch_sizes, cpu_and_maybe_cuda


@pytest.fixture
def project_root():
    return pathlib.Path(__file__).parent.parent.resolve()


@pytest.fixture(scope="module")
def media(project_root: pathlib.Path):
    return project_root / "media"


@pytest.fixture(scope="module")
def notebooks(project_root: pathlib.Path):
    return project_root / "notebooks"


@pytest.fixture(autouse=True, scope="session")
def enable_traceback():
    """
    Enable rich traceback for all tests.
    """
    traceback.install(show_locals=True, word_wrap=True)


@pytest.fixture(autouse=True, scope="session")
def seed():
    seed = 42
    random.seed(seed)
    npr.seed(seed)
    torch.manual_seed(seed)
    return seed


@pytest.fixture(params=cpu_and_maybe_cuda())
def device(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def data_size() -> int:
    return max(batch_sizes())


@pytest.fixture(params=batch_sizes())
def batch_size(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture
def fake_mode():
    with fake_fn():
        yield


@pytest.fixture
def real_mode():
    with track_fn():
        yield


@pytest.fixture(params=[fake_mode.name, real_mode.name])
def maybe_fake_mode(request: pytest.FixtureRequest):
    yield request.getfixturevalue(request.param)
