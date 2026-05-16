# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway.tags import check_dim_tag


def _valid_tags():
    yield "pippi"
    yield "tspi"
    yield "tppix"


def _invalid_tags():
    yield "hello"
    yield "ts pi"
    yield "tax"


@pytest.fixture(params=_valid_tags())
def valid_tags(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=_invalid_tags())
def invalid_tags(request: pytest.FixtureRequest):
    return request.param


def test_valid_tags(valid_tags: str):
    assert check_dim_tag(valid_tags)


def test_invalid_tags(invalid_tags: str):
    assert not check_dim_tag(invalid_tags)
