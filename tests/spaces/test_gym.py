# Copyright (c) AIoWay Authors - All Rights Reserved


import pytest
import torch
from torchrl.data import tensor_specs as tspecs

from aioway.spaces import array_box_spec


@pytest.fixture
def box():
    return array_box_spec(low=[0, 0], high=[1, 1], dtype=torch.float32)


def test_box_valid(box: tspecs.TensorSpec):
    assert box.contains(torch.tensor([[0.5, 0.2], [0.1, 0.9]]).float())
