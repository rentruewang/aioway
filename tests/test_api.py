# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from aioway import api


@pytest.fixture(scope="module", autouse=True)
def import_modules():
    import aioway._ufuncs as _


def test_api_not_empty():
    assert dir(api)


def test_api_items():
    for name in dir(api):
        item = getattr(api, name)
        assert item
