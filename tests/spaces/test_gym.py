# Copyright (c) AIoWay Authors - All Rights Reserved


import pytest
import torch

from aioway._torch import Shape
from aioway.spaces import (
    BoxSpace,
    DiscreteSpace,
    MultiBinarySpace,
    MultiDiscreteSpace,
    Space,
)


@pytest.fixture
def discrete():
    return DiscreteSpace(n=4)


@pytest.fixture
def box():
    return BoxSpace(low=torch.zeros(2), high=torch.ones(2))


@pytest.fixture
def multi_discrete():
    return MultiDiscreteSpace(nvec=torch.tensor([2, 3, 4]))


@pytest.fixture
def multi_binary():
    return MultiBinarySpace(shape=Shape.parse(2, 3))


def test_discrete_valid(discrete: Space):
    assert torch.tensor([2]) in discrete
    assert torch.tensor([1, 2]) in discrete


def test_discrete_valid_sample(discrete: Space):
    assert discrete.sample() in discrete


def test_discrete_invalid(discrete: Space):
    assert torch.tensor(4.0) not in discrete
    assert torch.tensor(-1) not in discrete


def test_discrete_invalid_init():
    with pytest.raises(ValueError):
        DiscreteSpace(n=0)


def test_box_valid(box: Space):
    assert torch.tensor([[0.5, 0.2], [0.1, 0.9]]) in box


def test_box_valid_sample(box: Space):
    assert box.sample() in box


def test_box_invalid(box: Space):
    assert torch.tensor([1.5, 0.2]) not in box
    assert torch.tensor([[0.5, 1.2]]) not in box
    assert torch.tensor([0.5]) not in box


def test_box_invalid_init():
    with pytest.raises(ValueError):
        BoxSpace(
            low=torch.tensor([1.0]),
            high=torch.tensor([0.0]),
        )


def test_multi_discrete_valid(multi_discrete: Space):
    assert torch.tensor([[1, 2, 3], [0, 1, 2]]) in multi_discrete


def test_multi_discrete_valid_sample(multi_discrete: Space):
    assert multi_discrete.sample() in multi_discrete


def test_multi_discrete_invalid(multi_discrete: Space):
    assert torch.tensor([1, 2, 3]) not in multi_discrete
    assert torch.tensor([2, 0, 0]) not in multi_discrete
    assert torch.tensor([1, 2]) not in multi_discrete
    assert torch.tensor([1.0, 2.0, 3.0]) not in multi_discrete


def test_multi_discrete_invalid_init():
    with pytest.raises(ValueError):
        MultiDiscreteSpace(nvec=torch.tensor([2, 0]))


def test_multi_binary_valid(multi_binary: Space):
    assert torch.tensor([[[0, 1, 0], [1, 0, 1]]]) in multi_binary


def test_multi_binary_valid_sample(multi_binary: Space):
    assert multi_binary.sample() in multi_binary


def test_multi_binary_invalid(multi_binary: Space):
    assert torch.tensor([[0, 1, 0], [1, 1, 0]]) not in multi_binary
    assert torch.tensor([0, 1, 0]) not in multi_binary


def test_multi_binary_invalid_init():
    with pytest.raises(ValueError):
        MultiBinarySpace(shape=Shape.parse())
