# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway import api


@pytest.fixture(scope="module", autouse=True)
def import_modules():
    pass


def test_api_items():
    for name in dir(api):
        item = getattr(api, name)
        assert item
