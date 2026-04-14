# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway.fn import track_fn_mode

import torch, pytest


@pytest.fixture
def a():
    return torch.randn(3)


@pytest.fixture
def b():
    return torch.randn(5)


def test_einsum(a: torch.Tensor, b: torch.Tensor):
    with track_fn_mode() as calls:
        result = torch.einsum("i,j->", a, b)

    assert result.ndim == 0
    assert len(calls)
