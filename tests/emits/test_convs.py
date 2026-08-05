# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch import nn

from aioway._torch import Shape
from aioway.emits import emit_one
from aioway.emits.convs import ImageRegressorEmitter
from aioway.spaces import FloatImageSpace, ShapeSpace


@pytest.fixture
def image_emitter():
    with ImageRegressorEmitter([1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4]).consider():
        yield


@pytest.fixture
def image_space():
    return FloatImageSpace(3)


@pytest.fixture
def output_space():
    return ShapeSpace(Shape.parse(3))


def test_emit_image_regressor(image_emitter, image_space, output_space):
    image_mod = emit_one(image_space, output_space)
    assert isinstance(image_mod, nn.Module)

    img = image_space.sample(7)
    assert image_mod(img).shape == (7, 3)
