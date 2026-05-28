# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib

import pytest
import torch

from aioway.io._media import ImageLoader, PillowImageLoader, TvioImageLoader
from aioway.tags import IsImageTag, extract_tags


def _loaders():
    yield TvioImageLoader()
    yield PillowImageLoader()


@pytest.fixture(params=_loaders(), ids=lambda l: type(l).__name__)
def loader(request: pytest.FixtureRequest):
    return request.param


def _read_image(example_image: pathlib.Path, loader: ImageLoader) -> torch.Tensor:
    image = loader(example_image).to_tensor()
    assert isinstance(image, torch.Tensor)
    assert image.dtype == torch.uint8
    return image


def test_read_image(example_image: pathlib.Path, loader: ImageLoader, maybe_fake_mode):
    _ = _read_image(example_image, loader)


def test_read_image_tags(
    example_image: pathlib.Path, loader: ImageLoader, maybe_fake_mode
):
    image = _read_image(example_image, loader)
    tags = extract_tags(image)
    assert IsImageTag.TAG in tags
    assert isinstance(tags[IsImageTag.TAG], IsImageTag)
