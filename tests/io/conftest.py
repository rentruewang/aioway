# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib

import pytest

from aioway.fn import fake_fn, track_fn


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


@pytest.fixture(params=[fake_fn, track_fn])
def maybe_fake_mode(request: pytest.FixtureRequest):
    with request.param() as hists:
        yield hists
