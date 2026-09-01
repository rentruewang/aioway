# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch import nn

from aioway.instrs import BaseLossInstr


@pytest.fixture(params=BaseLossInstr.__subclasses__())
def loss_instr(request):
    return request.param


@pytest.fixture
def loss_module(loss_instr: type[BaseLossInstr]):
    return loss_instr().module()


def test_loss_instr_to_module(loss_module: nn.Module):
    assert isinstance(loss_module, nn.Module)
