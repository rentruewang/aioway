# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib

import pytest
import torch

from aioway.fn.ctx import is_fake_tensor
from aioway.io import (
    ImageLoader,
    PillowImageLoader,
    TvioImageLoader,
)
from aioway.io.images import FakePillowImageLoader


@pytest.fixture
def torchvision_io_loader():
    return TvioImageLoader()


@pytest.fixture
def pillow_image_loader():
    return PillowImageLoader()


@pytest.fixture(params=[torchvision_io_loader.name, pillow_image_loader.name])
def image_loader(request: pytest.FixtureRequest):
    return request.getfixturevalue(request.param)


@pytest.fixture
def fake_pillow_image_loader():
    return FakePillowImageLoader()


def test_read_image(
    example_image: pathlib.Path, image_loader: ImageLoader, maybe_fake_mode
):
    image = image_loader(example_image)
    assert isinstance(image, torch.Tensor)
    assert image.dtype == torch.uint8


def test_read_image_fake(
    example_image: pathlib.Path, fake_pillow_image_loader: ImageLoader, fake_mode
):
    image = fake_pillow_image_loader(example_image)
    assert isinstance(image, torch.Tensor)
    assert is_fake_tensor(image)
    assert image.dtype == torch.uint8
