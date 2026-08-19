# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway.tspecs import TSpec, array_box_tspec


@pytest.fixture
def box():
    return array_box_tspec(low=[0, 0], high=[1, 1], dtype=torch.float32)


def test_box_valid(box: TSpec):
    assert box.contains(torch.tensor([[0.5, 0.2], [0.1, 0.9]]).float())
