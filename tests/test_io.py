# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest, pathlib

_MEDIA_ROOT = pathlib.Path(__file__).parent.parent / "media"


@pytest.fixture(scope="session")
def media():
    return _MEDIA_ROOT


def _example_files():
    yield _MEDIA_ROOT / "file_example_JPG_2500kB.jpg"
    yield _MEDIA_ROOT / "file_example_MP3_5MG.mp3"
    yield _MEDIA_ROOT / "file_example_MP4_1920_18MG.mp4"
    yield _MEDIA_ROOT / "file_example_PNG_3MB.png"


@pytest.fixture(params=_example_files())
def example_file(request: pytest.FixtureRequest):
    return request.param


def test_media_exits(media: pathlib.Path):
    assert media.exists()
    assert media.is_dir()


def test_example_file_exists(example_file: pathlib.Path):
    assert example_file.exists()
    assert example_file.is_file()
