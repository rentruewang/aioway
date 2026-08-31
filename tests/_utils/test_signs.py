# Copyright (c) AIoWay Authors - All Rights Reserved


import pytest

from aioway._utils import Sign


def int_add(a: int, b: int):
    return a + b


def float_add(a: float, b: float):
    return a + b


@pytest.fixture
def int_add_sign():
    return Sign.from_callable(int_add)


@pytest.fixture
def float_add_sign():
    return Sign.from_callable(float_add)


def test_sign_not_equal(int_add_sign, float_add_sign):
    assert int_add_sign != float_add_sign


def test_sign_outline_equal(int_add_sign: Sign, float_add_sign: Sign):
    assert int_add_sign.outline == float_add_sign.outline
