# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway import api


def test_api_not_empty():
    assert dir(api)


def test_api_items():
    for name in dir(api):
        item = getattr(api, name)
        assert item
