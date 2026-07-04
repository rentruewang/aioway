# Copyright (c) AIoWay Authors - All Rights Reserved


import pytest
import torch

from aioway.attrs import Shape
from aioway.spaces import BoxSpace, DiscreteSpace, MultiBinarySpace, MultiDiscreteSpace


@pytest.fixture
def discrete():
    return DiscreteSpace(n=4)


@pytest.fixture
def box():
    return BoxSpace(low=torch.zeros(2), high=torch.ones(2))


@pytest.fixture
def multidiscrete():
    return MultiDiscreteSpace(nvec=torch.tensor([2, 3, 4]))


@pytest.fixture
def multibinary():
    return MultiBinarySpace(shape=Shape.parse(2, 3))


def test_discrete_valid(discrete):
    assert torch.tensor([2]) in discrete
    assert torch.tensor([1, 2]) in discrete


def test_discrete_invalid(discrete):
    assert torch.tensor(4.0) not in discrete
    assert torch.tensor(-1) not in discrete


def test_discrete_invalid_init():
    with pytest.raises(ValueError):
        DiscreteSpace(n=0)


def test_box_valid(box):
    assert torch.tensor([[0.5, 0.2], [0.1, 0.9]]) in box


def test_box_invalid(box):
    assert torch.tensor([1.5, 0.2]) not in box
    assert torch.tensor([[0.5, 1.2]]) not in box
    assert torch.tensor([0.5]) not in box


def test_box_invalid_init():
    with pytest.raises(ValueError):
        BoxSpace(
            low=torch.tensor([1.0]),
            high=torch.tensor([0.0]),
        )


def test_multidiscrete_valid(multidiscrete):
    assert torch.tensor([[1, 2, 3], [0, 1, 2]]) in multidiscrete


def test_multidiscrete_invalid(multidiscrete):
    assert torch.tensor([1, 2, 3]) not in multidiscrete
    assert torch.tensor([2, 0, 0]) not in multidiscrete
    assert torch.tensor([1, 2]) not in multidiscrete
    assert torch.tensor([1.0, 2.0, 3.0]) not in multidiscrete


def test_multidiscrete_invalid_init():
    with pytest.raises(ValueError):
        MultiDiscreteSpace(nvec=torch.tensor([2, 0]))


def test_multibinary_valid(multibinary):
    assert torch.tensor([[[0, 1, 0], [1, 0, 1]]]) in multibinary  # batch


def test_multibinary_invalid(multibinary):
    assert torch.tensor([[0, 1, 0], [1, 1, 0]]) not in multibinary
    assert torch.tensor([0, 1, 0]) not in multibinary


def test_multibinary_invalid_init():
    with pytest.raises(ValueError):
        MultiBinarySpace(shape=Shape.parse())
