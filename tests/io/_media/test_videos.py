# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib

import pytest
import torch

from aioway.dsets import AvVideoLoader, TorchCodecVideoLoader, VideoLoader
from aioway.tags import IsVideoTag


def _loader():
    yield AvVideoLoader()
    yield TorchCodecVideoLoader()


@pytest.fixture(params=_loader(), ids=lambda x: type(x).__name__)
def loader(request: pytest.FixtureRequest):
    return request.param


def test_media_exits(media: pathlib.Path):
    assert media.exists()
    assert media.is_dir()


def test_example_file_exists(example_file: pathlib.Path):
    assert example_file.exists()
    assert example_file.is_file()


def _read_video(video: pathlib.Path, loader: VideoLoader):
    result = loader(video)
    return result.to_tensor()


def test_read_video(example_video: pathlib.Path, loader: VideoLoader, maybe_fake_mode):
    video = _read_video(example_video, loader)
    assert isinstance(video, torch.Tensor)


def test_read_video_tags(
    example_video: pathlib.Path, loader: VideoLoader, maybe_fake_mode
):
    video = _read_video(example_video, loader)
    assert isinstance(video, torch.Tensor)
    assert IsVideoTag.extract(video) is not None
