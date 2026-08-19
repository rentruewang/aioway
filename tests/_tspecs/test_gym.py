# Copyright (c) AIoWay Authors - All Rights Reserved


import pytest
import torch

from aioway._tspecs import array_box_space
from aioway._tspecs.spaces import TSpecSpace


@pytest.fixture
def box():
    return array_box_space(low=[0, 0], high=[1, 1], dtype=torch.float32)


def test_box_valid(box: TSpecSpace):
    assert torch.tensor([[0.5, 0.2], [0.1, 0.9]]).float() in box
