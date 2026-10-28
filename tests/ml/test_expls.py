# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn


@pytest.fixture
def linear():
    return nn.Linear(4, 1)


@pytest.fixture
def input():
    return torch.randn(32, 4)
