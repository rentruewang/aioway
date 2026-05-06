# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib

import pytest
import torch

from aioway.fn.routers import fake_fn, track_fn
from aioway.io import read_audio_from_path, read_image_from_path, read_video_from_path


@pytest.fixture(scope="session")
def media():
    return pathlib.Path(__file__).parent.parent / "media"


@pytest.fixture(params=[fake_fn, track_fn], autouse=True)
def maybe_fake_mode(request: pytest.FixtureRequest):
    with request.param() as hists:
        yield hists


@pytest.fixture
def example_jpg(media: pathlib.Path):
    return media / "file_example_JPG_2500kB.jpg"


@pytest.fixture
def example_png(media: pathlib.Path):
    return media / "file_example_PNG_3MB.png"


@pytest.fixture
def example_mp4(media: pathlib.Path):
    return media / "file_example_MP4_1920_18MG.mp4"


@pytest.fixture
def example_mp3(media: pathlib.Path):
    return media / "file_example_MP3_5MG.mp3"


@pytest.fixture(
    params=[example_jpg.name, example_png.name, example_mp3.name, example_mp4.name]
)
def example_file(request: pytest.FixtureRequest):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=[example_jpg.name, example_png.name])
def example_image(request: pytest.FixtureRequest):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=[example_mp3.name, example_mp4.name])
def example_audio(request: pytest.FixtureRequest):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=[example_mp4.name])
def example_video(request: pytest.FixtureRequest):
    return request.getfixturevalue(request.param)


def test_media_exits(media: pathlib.Path):
    assert media.exists()
    assert media.is_dir()


def test_example_file_exists(example_file: pathlib.Path):
    assert example_file.exists()
    assert example_file.is_file()


def test_read_img(example_image: pathlib.Path):
    img = read_image_from_path(example_image)
    assert isinstance(img, torch.Tensor)


def test_read_audio(example_audio: pathlib.Path):
    audio = read_audio_from_path(example_audio)
    assert isinstance(audio.data, torch.Tensor)


def test_read_video(example_video: pathlib.Path):
    video = read_video_from_path(example_video)
    assert isinstance(video, torch.Tensor)
