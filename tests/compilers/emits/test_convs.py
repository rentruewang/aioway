# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch import nn

from aioway.compilers import emit_one
from aioway.compilers.emits.convs import ImageRegressorEmitter
from aioway.schemas import Shape
from aioway.tspecs import (
    TSpec,
    float_image_tspec,
    sample_from_tspec,
    set_batch_size,
    unbounded_box_tspec,
)


@pytest.fixture
def image_emitter():
    with (
        set_batch_size(4),
        ImageRegressorEmitter([5, 6, 7, 8], [1, 2, 3, 4], [1, 2, 3, 4]).consider(),
    ):
        yield


@pytest.fixture
def image_tspec():
    return float_image_tspec(3, 28, 28)


@pytest.fixture(params=[3, 5, 1000])
def feat_size(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture
def output_tspec(feat_size: int):
    return unbounded_box_tspec(Shape.parse(feat_size))


def test_emit_image_regressor(
    image_emitter,
    image_tspec: TSpec,
    output_tspec: TSpec,
    feat_size: int,
):
    image_mod = emit_one(image_tspec, output_tspec)
    assert isinstance(image_mod, nn.Module)

    with set_batch_size(7):
        img = sample_from_tspec(image_tspec)
    assert image_mod(img).shape == (7, feat_size)
