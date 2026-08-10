# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch import nn
from torchrl import data as rldata

from aioway._specs import float_image, unbounded_box_spec
from aioway._torch import Shape
from aioway.emits import emit_one
from aioway.emits.convs import ImageRegressorEmitter


@pytest.fixture
def image_emitter():
    with ImageRegressorEmitter([1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4]).consider():
        yield


@pytest.fixture
def image_space():
    return float_image(3)


@pytest.fixture(params=[3, 5, 1000])
def feat_size(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture
def output_space(feat_size: int):
    return unbounded_box_spec(Shape.parse(feat_size))


def test_emit_image_regressor(
    image_emitter,
    image_space: rldata.TensorSpec,
    output_space: rldata.TensorSpec,
    feat_size: int,
):
    image_mod = emit_one(image_space, output_space)
    assert isinstance(image_mod, nn.Module)

    img = image_space.sample(7)
    assert image_mod(img).shape == (7, feat_size)
