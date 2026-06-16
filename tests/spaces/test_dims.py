# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls

import pytest
import torch

from aioway.spaces import DimSpace


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
    _ = DimSpace(valid_tags.tag)


def test_invalid_tags(invalid_tags: TensorAndTag):
    with pytest.raises(ValueError):
        tag = DimSpace(invalid_tags.tag)

        if invalid_tags.tensor not in tag:
            raise ValueError
