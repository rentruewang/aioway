# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib
import random
import typing
from collections import abc as cabc

import git
import pytest
import torch
from numpy import random as npr
from rich import traceback

from aioway.fake import fake_fn, track_fn

from .fake import batch_sizes, cpu_and_maybe_cuda

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve()
_REPO = git.Repo(_PROJECT_ROOT)
_MEDIA_DIR = _PROJECT_ROOT / "media"
_NOTEBOOKS_DIR = _PROJECT_ROOT / "notebooks"
_SRC_DIR = _PROJECT_ROOT / "src"
_TESTS_DIR = _PROJECT_ROOT / "tests"
_GITIGNORE = _PROJECT_ROOT / ".gitignore"


@typing.no_type_check
def _file_paths():
    for item in _REPO.tree().traverse():
        if item.type != "blob":
            continue

        yield item.path

    for path in _REPO.untracked_files:
        yield path

    for item in _REPO.index.diff("HEAD"):
        a_path = item.a_path
        assert a_path
        yield a_path


def file_paths() -> cabc.Generator[pathlib.Path]:
    "The file paths that are in this repo, tracked or not."

    for fp in _file_paths():
        if (path := pathlib.Path(fp).resolve()).exists():
            yield path


def git_files(folder: pathlib.Path) -> cabc.Generator[pathlib.Path]:
    "The git files under `folder`."
    files = set(file_paths())

    for path in files:
        if path.is_relative_to(folder):
            yield path


def python_git_files(folder: pathlib.Path) -> cabc.Generator[pathlib.Path]:
    "The git files that are python."

    for file in git_files(folder):
        if file.suffix == ".py":
            yield file


def _relative_to_root(path: pathlib.Path) -> pathlib.Path:
    return path.relative_to(_PROJECT_ROOT)


@pytest.fixture(scope="module")
def project_root():
    assert _PROJECT_ROOT.exists()
    assert _PROJECT_ROOT.is_dir()
    return _PROJECT_ROOT


@pytest.fixture(scope="module")
def media():
    assert _MEDIA_DIR.exists()
    assert _MEDIA_DIR.is_dir()
    return _MEDIA_DIR


def _notebooks():
    assert _NOTEBOOKS_DIR.exists()
    assert _NOTEBOOKS_DIR.is_dir()

    yield from python_git_files(_NOTEBOOKS_DIR)


@pytest.fixture(scope="module", params=_notebooks(), ids=lambda path: path.name)
def notebook(request: pytest.FixtureRequest):
    return request.param


def _src_tests():
    yield from python_git_files(_SRC_DIR)
    yield from python_git_files(_TESTS_DIR)


@pytest.fixture(
    scope="module",
    params=_src_tests(),
    ids=lambda path: str(_relative_to_root(path)),
)
def src_test_py(request: pytest.FixtureRequest):
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
