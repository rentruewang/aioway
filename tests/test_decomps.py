# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls

import pytest
import torch

from aioway.decomps import find_nested_tensors


@pytest.fixture
def a():
    return torch.tensor([3])


@pytest.fixture
def b():
    return torch.tensor([4])


@pytest.fixture
def c():
    return torch.tensor([5, 6, 7])


@pytest.fixture
def d():
    return torch.tensor([8, 9])


@dcls.dataclass(frozen=True)
class NestedTensors:
    lists: list[torch.Tensor]
    dicts: dict[str, list[torch.Tensor]]


@pytest.fixture
def nested(a, b, c, d):
    return NestedTensors([a], {"b": [b], "cd": [c, d]})


def test_find_nested_tensors(nested, a, b, c, d):
    result = set(find_nested_tensors(nested))
    assert result == {a, b, c, d}
