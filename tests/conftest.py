# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib
import random

import pathspec
import pytest
import torch
from numpy import random as npr
from rich import traceback

from aioway.fn import fake_fn, track_fn

from .fake import batch_sizes, cpu_and_maybe_cuda

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve()
MEDIA_DIR = PROJECT_ROOT / "media"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
GITIGNORE = PROJECT_ROOT / ".gitignore"


def gitignore_glob(path: pathlib.Path, pattern: str):
    assert GITIGNORE.exists()
    spec = pathspec.PathSpec.from_lines("gitwildmatch", GITIGNORE.open("r"))

    for f in path.rglob(pattern):
        if not spec.match_file(f):
            yield f


@pytest.fixture(scope="module")
def project_root():
    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.is_dir()
    return PROJECT_ROOT


@pytest.fixture(scope="module")
def media():
    assert MEDIA_DIR.exists()
    assert MEDIA_DIR.is_dir()
    return MEDIA_DIR


def _notebooks():
    assert NOTEBOOKS_DIR.exists()
    assert NOTEBOOKS_DIR.is_dir()
    yield from gitignore_glob(NOTEBOOKS_DIR, "*.py")


@pytest.fixture(scope="module", params=_notebooks())
def notebook(request: pytest.FixtureRequest):
    return request.param


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
