# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway.spaces import ByteImageSpace, FloatImageSpace, ImageSpace, coerce_space


@pytest.fixture
def byte_img():
    return ByteImageSpace(3)


@pytest.fixture
def float_img():
    return FloatImageSpace(3)


@pytest.fixture(params=[byte_img.name, float_img.name])
def img(request: pytest.FixtureRequest):
    return request.getfixturevalue(request.param)


def test_coerce(img: ImageSpace):
    assert isinstance(img, ImageSpace)
    assert coerce_space(img, ImageSpace).out_space is img
