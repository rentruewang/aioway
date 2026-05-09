# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib
import random

import pytest
import torch
from numpy import random as npr
from rich import traceback

from .fake import batch_sizes, cpu_and_maybe_cuda


@pytest.fixture(scope="module")
def media():
    return (pathlib.Path(__file__).parent.parent / "media").resolve()


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
