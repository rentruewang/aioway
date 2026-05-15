# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import tensordict as td
import torch
from numpy import random as np_rand

from aioway.schemas import Attr, AttrSet
from tests.fake import batch_sizes, chunk_ok, cpu_and_maybe_cuda


@pytest.fixture(params=cpu_and_maybe_cuda(), scope="session")
def device(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=batch_sizes(), scope="module")
def batch(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture
def chunk(device: str, batch: int) -> td.TensorDict:
    return chunk_ok(device=device, size=batch)


@pytest.fixture
def schema():
    return AttrSet.from_values(
        f1d=Attr.parse(
            device="cpu",
            shape=[1],
            dtype="float32",
            layout="strided",
            requires_grad=False,
        ),
        f2d=Attr.parse(
            device="cpu",
            shape=[1, 32],
            dtype="float32",
            layout="strided",
            requires_grad=False,
        ),
        i1d=Attr.parse(
            device="cpu",
            shape=[1],
            dtype="int64",
            layout="strided",
            requires_grad=False,
        ),
        i2d=Attr.parse(
            device="cpu",
            shape=[1, 32],
            dtype="int64",
            layout="strided",
            requires_grad=False,
        ),
    )


def test_chunk_init_success(chunk: td.TensorDict) -> None:
    _ = chunk


def test_chunk_len(batch: int, chunk: td.TensorDict) -> None:
    assert len(chunk) == batch


def test_chunk_getitem_size(batch: int, chunk: td.TensorDict) -> None:

    assert len(chunk[batch - 1 : batch]) == 1
    assert len(chunk[[0]]) == 1
    assert len(chunk[[-1]]) == 1

    # Bool index in torch.
    torch_idx = torch.randn(batch) > 0
    assert len(chunk[torch_idx]) == (torch_idx > 0).sum()

    # Int index in torch.
    indexed = chunk[torch.arange(len(torch_idx))[torch_idx]]
    assert len(indexed) == (torch_idx > 0).sum().item()

    # Bool index in numpy.
    np_idx = np_rand.randn(batch) < 0
    assert len(chunk[torch.tensor(np_idx)]) == np_idx.sum()


def test_chunk_keys(device: str, batch: int, chunk: td.TensorDict) -> None:
    assert set(chunk.keys()) == {"f1d", "f2d", "i1d", "i2d"}
