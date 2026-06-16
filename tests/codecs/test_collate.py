# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
import torch

from aioway.codecs import chunk_collate_decompose_dim


class SamplesCollateDim(typing.NamedTuple):
    samples: list[torch.Tensor]
    collate_dim: int


@pytest.fixture
def max_len():
    return 100


def _random_tensor_lists():
    yield SamplesCollateDim(
        [torch.randn(1001, 33), torch.randn(1002, 33), torch.randn(101, 33)],
        0,
    )

    yield SamplesCollateDim(
        [
            torch.randn(1001, 2, 71),
            torch.randn(1002, 2, 71),
            torch.randn(101, 2, 71),
        ],
        0,
    )

    yield SamplesCollateDim(
        [
            torch.randn(2, 1001, 71),
            torch.randn(2, 1002, 71),
            torch.randn(2, 101, 71),
        ],
        1,
    )

    yield SamplesCollateDim(
        [
            torch.randn(2, 71, 1002),
            torch.randn(2, 71, 1001),
            torch.randn(2, 71, 1101),
        ],
        -1,
    )


@pytest.fixture(params=_random_tensor_lists())
def samples(request: pytest.FixtureRequest):
    return request.param


def test_chunk_collate(samples: SamplesCollateDim, max_len: int):
    tensor_list, collate_dim = samples
    [ndim] = {t.ndim for t in tensor_list}
    collator = chunk_collate_decompose_dim(max_len, collate_dim)
    result = collator(tensor_list)

    # After decompose, first dim is batch, rest is unchanged relatively.
    # Also we need collate_dim to be positive here.
    assert result.shape[collate_dim % ndim + 1] == max_len

    # Adds the batch dimension
    assert result.ndim == ndim + 1
