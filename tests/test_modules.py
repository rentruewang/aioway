# Copyright (c) AIoWay Authors - All Rights Reserved

from collections import abc as cabc

import pytest
import torch
from torch import nn

from aioway._modules import rebuild_module
from aioway._torch import (
    fake_mode,
    is_fake_tensor,
    is_real_tensor,
    real_mode,
)


@pytest.fixture
def fake_module() -> nn.Module:
    with fake_mode():
        return _linear()


def _linear() -> nn.Module:
    return nn.Sequential(nn.Linear(3, 5), nn.Linear(5, 7))


def _params_and_buffers(module: nn.Module) -> cabc.Generator[torch.Tensor]:
    yield from module.parameters()
    yield from module.buffers()


def test_module_is_fake(fake_module: nn.Module):
    assert all(map(is_fake_tensor, _params_and_buffers(fake_module)))


def test_module_rebuild(fake_module: nn.Module):
    assert all(map(is_fake_tensor, _params_and_buffers(fake_module)))

    with real_mode():
        module = rebuild_module(fake_module)

    assert all(map(is_real_tensor, _params_and_buffers(module)))
