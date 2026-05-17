# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
import torch

from aioway.tags.dims import DimTag


def _valid_tags():
    yield "pippi", torch.randn(3, 3, 3, 3, 3)
    yield "tspi", torch.randn(3, 3, 3, 3)
    yield "tppix", torch.randn(3, 3, 3, 3, 3)


def _invalid_tags():
    # Invalid tag.
    yield "hello", torch.randn(3, 3, 3, 3, 3)
    yield "ts pi", torch.randn(3, 3, 3, 3, 3)
    yield "tax", torch.randn(3, 3, 3)

    # Dim mismatch
    yield "tspi", torch.randn(3, 3)
    yield "tppix", torch.randn(3, 3)


class TensorAndTag(typing.NamedTuple):
    tag: str
    tensor: torch.Tensor


@pytest.fixture(params=_valid_tags())
def valid_tags(request: pytest.FixtureRequest):
    return TensorAndTag(*request.param)


@pytest.fixture(params=_invalid_tags())
def invalid_tags(request: pytest.FixtureRequest):
    return TensorAndTag(*request.param)


def test_valid_tags(valid_tags: TensorAndTag):
    assert DimTag(valid_tags.tensor, valid_tags.tag)


def test_invalid_tags(invalid_tags: TensorAndTag):
    with pytest.raises(ValueError):
        assert DimTag(invalid_tags.tensor, invalid_tags.tag)
