# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib

import torch

from aioway.io import read_video_from_path


def test_media_exits(media: pathlib.Path):
    assert media.exists()
    assert media.is_dir()


def test_example_file_exists(example_file: pathlib.Path):
    assert example_file.exists()
    assert example_file.is_file()


def test_read_video(example_video: pathlib.Path, maybe_fake_mode):
    video = read_video_from_path(example_video)
    assert isinstance(video, torch.Tensor)
