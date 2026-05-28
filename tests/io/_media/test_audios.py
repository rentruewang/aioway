# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib

import pytest
import torch

from aioway.io._media import (
    AudioLoader,
    AvAudioLoader,
    TorchCodecAudioLoader,
    encode_with_stft,
)
from aioway.tags import SampleRateTag


def _loaders():
    yield TorchCodecAudioLoader()
    yield AvAudioLoader()


@pytest.fixture(params=_loaders(), ids=lambda l: type(l).__name__)
def loader(request: pytest.FixtureRequest):
    return request.param


def _read_audio(audio: pathlib.Path, loader: AudioLoader):
    result = loader(audio)
    return result.to_tensor()


def test_read_audio(example_audio: pathlib.Path, loader: AudioLoader, maybe_fake_mode):
    audio = _read_audio(example_audio, loader)
    assert isinstance(audio, torch.Tensor)


def test_read_audio_tags(
    example_audio: pathlib.Path, loader: AudioLoader, maybe_fake_mode
):
    audio = _read_audio(example_audio, loader)
    assert SampleRateTag.extract(audio) is not None


def test_read_audio_stft(
    example_audio: pathlib.Path, loader: AudioLoader, maybe_fake_mode
):
    audio = _read_audio(example_audio, loader)
    stft = encode_with_stft(audio, 20)
    assert isinstance(stft, torch.Tensor)
    assert SampleRateTag.extract(stft) == SampleRateTag.extract(audio)
