# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib

import pytest
import torch

from aioway.io import AudioLoader, AvAudioLoader, TorchCodecAudioLoader
from aioway.tags import SampleRateTag


def _loaders():
    yield TorchCodecAudioLoader()
    yield AvAudioLoader()


@pytest.fixture(params=_loaders())
def loader(request: pytest.FixtureRequest):
    return request.param


def test_read_audio(example_audio: pathlib.Path, loader: AudioLoader, maybe_fake_mode):
    audio = loader(example_audio)
    assert isinstance(audio, torch.Tensor)


def test_read_audio_tags(
    example_audio: pathlib.Path, loader: AudioLoader, maybe_fake_mode
):
    audio = loader(example_audio)
    assert SampleRateTag.extract(audio) is not None
