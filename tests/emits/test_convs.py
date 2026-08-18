# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch import nn
from torchrl.data import tensor_specs as tspecs

from aioway._torch import Shape
from aioway.nets import emit_one, sample_from_spec, set_batch_size
from aioway.nets.convs import ImageRegressorEmitter
from aioway.spaces import float_image_spec, unbounded_box_space


@pytest.fixture
def image_emitter():
    with (
        set_batch_size(4),
        ImageRegressorEmitter([5, 6, 7, 8], [1, 2, 3, 4], [1, 2, 3, 4]).consider(),
    ):
        yield


@pytest.fixture
def image_space():
    return float_image_spec(3, 28, 28)


@pytest.fixture(params=[3, 5, 1000])
def feat_size(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture
def output_space(feat_size: int):
    return unbounded_box_space(Shape.parse(feat_size))


def test_emit_image_regressor(
    image_emitter,
    image_space: tspecs.TensorSpec,
    output_space: tspecs.TensorSpec,
    feat_size: int,
):
    image_mod = emit_one(image_space, output_space)
    assert isinstance(image_mod, nn.Module)

    with set_batch_size(7):
        img = sample_from_spec(image_space)
    assert image_mod(img).shape == (7, feat_size)
