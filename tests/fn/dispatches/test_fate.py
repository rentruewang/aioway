# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch


@pytest.fixture
def left():
    return torch.randn(3)


@pytest.fixture
def right():
    return torch.randn(3)
