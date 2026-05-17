# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib

import pytest
import torch

from aioway.io import AudioLoader, TorchCodecAudioLoader, read_video_from_path
from aioway.tags import SampleRateTag


def _loaders():
    yield TorchCodecAudioLoader()


@pytest.fixture(params=_loaders())
def loader(request: pytest.FixtureRequest):
    return request.param


def test_media_exits(media: pathlib.Path):
    assert media.exists()
    assert media.is_dir()


def test_example_file_exists(example_file: pathlib.Path):
    assert example_file.exists()
    assert example_file.is_file()


def test_read_audio(example_audio: pathlib.Path, loader: AudioLoader, maybe_fake_mode):
    audio = loader(example_audio)
    assert isinstance(audio, torch.Tensor)


def test_read_audio_tags(
    example_audio: pathlib.Path, loader: AudioLoader, maybe_fake_mode
):
    audio = loader(example_audio)
    assert SampleRateTag.extract(audio) is not None


def test_read_video(example_video: pathlib.Path, maybe_fake_mode):
    video = read_video_from_path(example_video)
    assert isinstance(video, torch.Tensor)
