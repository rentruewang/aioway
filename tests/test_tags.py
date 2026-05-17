# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
import torch

from aioway.tags import DimTag, extract_tags


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
    _ = DimTag(valid_tags.tensor, valid_tags.tag)


def test_invalid_tags(invalid_tags: TensorAndTag):
    with pytest.raises(ValueError):
        _ = DimTag(invalid_tags.tensor, invalid_tags.tag)


def test_attach(valid_tags: TensorAndTag):
    tag = DimTag(valid_tags.tensor, valid_tags.tag)
    assert tag.tensor is valid_tags.tensor
    assert hasattr(tag.tensor, DimTag.TAG)
    assert DimTag.extract(valid_tags.tensor) is tag
    assert extract_tags(valid_tags.tensor) == {DimTag.TAG: tag}


def test_tag_eq(valid_tags: TensorAndTag):
    tag = DimTag(valid_tags.tensor, valid_tags.tag)
    another = valid_tags.tensor.clone()

    assert not extract_tags(another)

    other_tag = tag.attach(another)

    assert extract_tags(another)
    assert tag is not other_tag
    assert tag == other_tag
