# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls

import pytest
import tensordict as td
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


@dcls.dataclass
class TensorAndTag:
    tag: str
    tensor: torch.Tensor

    def __post_init__(self):
        # Copy because the tensors were generated in the collect phase,
        # so if we don't copy they won't be reinitialized,
        # which messes with our tag detection logic
        self.tensor = self.tensor.clone()


@pytest.fixture(params=_valid_tags())
def valid_tags(request: pytest.FixtureRequest):
    return TensorAndTag(*request.param)


@pytest.fixture(params=_invalid_tags())
def invalid_tags(request: pytest.FixtureRequest):
    return TensorAndTag(*request.param)


def test_valid_tags(valid_tags: TensorAndTag):
    DimTag(valid_tags.tag).attach(valid_tags.tensor)


def test_invalid_tags(invalid_tags: TensorAndTag):
    with pytest.raises(ValueError):
        tag = DimTag(invalid_tags.tag)
        tag.attach(invalid_tags.tensor)


def test_attach(valid_tags: TensorAndTag):
    assert DimTag.extract(valid_tags.tensor) is None
    tag = DimTag(valid_tags.tag)
    tag.attach(valid_tags.tensor)
    assert hasattr(valid_tags.tensor, DimTag.TAG)
    assert DimTag.extract(valid_tags.tensor) is tag
    assert extract_tags(valid_tags.tensor) == {DimTag.TAG: tag}


def test_attach_preserve_after_tdict(valid_tags: TensorAndTag):
    assert DimTag.extract(valid_tags.tensor) is None
    tag = DimTag(valid_tags.tag)
    tag.attach(valid_tags.tensor)
    tdict = td.TensorDict({"a": valid_tags.tensor, "b": valid_tags.tensor})
    assert tdict["a"] is tdict["b"]
    assert DimTag.extract(tdict["a"]) is DimTag.extract(tdict["b"]) is tag


def test_tag_eq(valid_tags: TensorAndTag):
    tag = DimTag(valid_tags.tag)
    tag.attach(valid_tags.tensor)
    another = valid_tags.tensor.clone()

    assert not extract_tags(another)

    tag.attach(another)
    other_tag = DimTag.extract(another)

    assert extract_tags(another)
    assert tag is other_tag
