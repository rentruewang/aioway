# Copyright (c) AIoWay Authors - All Rights Reserved

import numpy as np
import pytest

from aioway.attrs import Shape


def _shapes():
    yield Shape.parse(3, 5, 7)
    yield Shape.parse([3, 5, 7])


@pytest.fixture(params=_shapes())
def shape(request: pytest.FixtureRequest) -> Shape:
    return request.param


def test_shape_getitem(shape: Shape):
    assert shape[0] == 3
    assert shape[1] == 5
    assert shape[2] == 7


def test_shape_size(shape: Shape):
    assert shape.numel() == 3 * 5 * 7


def test_shape_ndim(shape: Shape):
    assert shape.ndim == 3


def test_shape_no_fail(shape: Shape):
    "Ensure that shapes do not crash when the other type is not recognized."

    assert shape != object()


def test_shape_ndarray(shape: Shape):
    assert shape == np.array([3, 5, 7])
    assert shape != np.array([[3, 5, 7]])
