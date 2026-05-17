# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib

import pytest
import torch

from aioway.fake import is_fake_tensor
from aioway.io import (
    ImageLoader,
    PillowImageLoader,
    TvioImageLoader,
)
from aioway.io.images import FakePillowImageLoader
from aioway.tags import IsImageTag, extract_tags


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


def _read_image(example_image: pathlib.Path, image_loader: ImageLoader) -> torch.Tensor:
    image = image_loader(example_image)
    assert isinstance(image, torch.Tensor)
    assert image.dtype == torch.uint8
    return image


def test_read_image(
    example_image: pathlib.Path, image_loader: ImageLoader, maybe_fake_mode
):
    _ = _read_image(example_image, image_loader)


def test_read_image_tags(
    example_image: pathlib.Path, image_loader: ImageLoader, maybe_fake_mode
):
    image = _read_image(example_image, image_loader)
    tags = extract_tags(image)
    assert IsImageTag.TAG in tags
    assert isinstance(tags[IsImageTag.TAG], IsImageTag)


def test_read_image_fake(
    example_image: pathlib.Path, fake_pillow_image_loader: ImageLoader, fake_mode
):
    image = _read_image(example_image, fake_pillow_image_loader)
    assert is_fake_tensor(image)
