# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway.t.keys import THash, TKey


@pytest.fixture(
    params=[
        lambda: torch.arange(12.0).reshape(3, 4),
        lambda: torch.arange(6),
        lambda: torch.tensor([True, False, True]),
        lambda: torch.randn(2, 3).half(),
        lambda: torch.randn(2, 3, dtype=torch.float64),
        lambda: torch.randn(2, 3, dtype=torch.complex128),
        lambda: torch.tensor(3.0),
        lambda: torch.zeros(0, 2),
    ]
)
def tensor(request) -> torch.Tensor:
    return request.param()


class TestTKey:
    def test_same_content(self, tensor):
        other = tensor.clone()
        assert TKey(tensor) == TKey(other)
        assert hash(TKey(tensor)) == hash(TKey(other))

    def test_eq_tensor(self, tensor):
        assert TKey(tensor) == tensor.clone()

    def test_eq_other_type(self, tensor):
        assert TKey(tensor) != 1

    def test_different_content(self):
        a = torch.arange(6.0)
        assert TKey(a) != TKey(a + 1)

    def test_different_dtype(self):
        a = torch.arange(6)
        assert TKey(a) != TKey(a.float())

    def test_different_shape(self):
        a = torch.arange(6)
        assert TKey(a) != TKey(a.reshape(2, 3))

    def test_dedup(self):
        a = torch.arange(6.0)
        assert len({TKey(a), TKey(a.clone()), TKey(a + 1)}) == 2


class TestTHash:
    def test_same_content(self, tensor):
        assert THash()(tensor) == THash()(tensor.clone())

    def test_different_content(self):
        a = torch.arange(6.0)
        assert THash()(a) != THash()(a + 1)
