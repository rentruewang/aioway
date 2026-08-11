# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway import api
from aioway._api import route_fastapi


@pytest.fixture(autouse=True)
def skip_if_no_fastapi():
    try:
        pass
    except ImportError:
        pytest.xfail()


def test_register_fastapi():
    @route_fastapi("hi", "get")
    def hi():
        return 1

    assert hi() == 1


def test_api_items():
    for name in dir(api):
        item = getattr(api, name)
        assert item
