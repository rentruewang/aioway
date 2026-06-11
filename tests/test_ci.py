# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib


def test_copyright_header(src_test_py: pathlib.Path):
    with src_test_py.open("r") as f:
        first = f.readline().rstrip("\n")

    assert first == "# Copyright (c) AIoWay Authors - All Rights Reserved"
