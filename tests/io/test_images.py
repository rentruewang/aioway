# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib

import pytest
import torch

from aioway.io import (
    ImageLoader,
    PillowImageLoader,
    TvioImageLoader,
)


@pytest.fixture
def torchvision_io_loader():
    return TvioImageLoader()


@pytest.fixture
def pillow_image_loader():
    return PillowImageLoader()


@pytest.fixture(params=[torchvision_io_loader.name, pillow_image_loader.name])
def image_loader(request: pytest.FixtureRequest):
    return request.getfixturevalue(request.param)


def test_read_image(
    example_image: pathlib.Path, image_loader: ImageLoader, maybe_fake_mode
):
    image = image_loader(example_image)
    assert isinstance(image.data, torch.Tensor)
    assert image.dtype == torch.uint8
