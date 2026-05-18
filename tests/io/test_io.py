# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib


def test_media_exits(media: pathlib.Path):
    assert media.exists()
    assert media.is_dir()


def test_example_file_exists(example_file: pathlib.Path):
    assert example_file.exists()
    assert example_file.is_file()
