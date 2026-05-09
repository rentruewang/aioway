# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib

import torch

from aioway.io import read_audio_from_path, read_video_from_path


def test_media_exits(media: pathlib.Path):
    assert media.exists()
    assert media.is_dir()


def test_example_file_exists(example_file: pathlib.Path):
    assert example_file.exists()
    assert example_file.is_file()


def test_read_audio(example_audio: pathlib.Path, maybe_fake_mode):
    audio = read_audio_from_path(example_audio)
    assert isinstance(audio.data, torch.Tensor)


def test_read_video(example_video: pathlib.Path, maybe_fake_mode):
    video = read_video_from_path(example_video)
    assert isinstance(video, torch.Tensor)
