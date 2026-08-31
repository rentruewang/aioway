# Copyright (c) AIoWay Authors - All Rights Reserved


import pytest

from aioway._utils import Sign


def int_add(a: int, b: int):
    return a + b


def float_add(a: float, b: float):
    return a + b


def add(a, b):
    return a + b


@pytest.fixture
def add_sign_i():
    return Sign.from_callable(int_add)


@pytest.fixture
def add_sign_f():
    return Sign.from_callable(float_add)


@pytest.fixture
def add_sign():
    return Sign.from_callable(add)


def test_sign_not_equal(add_sign_i: Sign, add_sign_f: Sign, add_sign: Sign):
    assert add_sign_i != add_sign_f
    assert add_sign_i != add_sign
    assert add_sign_f != add_sign


def test_sign_outline_equal(add_sign_i: Sign, add_sign_f: Sign, add_sign: Sign):
    assert add_sign_i.strip_type() == add_sign_f.strip_type() == add_sign.strip_type()
