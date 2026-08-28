# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway.instrs import BaseLossInstr, NnLoss


@pytest.fixture(params=BaseLossInstr.__subclasses__())
def loss_instr(request):
    return request.param


@pytest.fixture
def loss_module(loss_instr: type[BaseLossInstr]):
    return loss_instr().module()


def test_loss_instr_to_module(loss_module: NnLoss):
    pass


def test_loss_instr_parse(loss_instr: type[BaseLossInstr], loss_module: NnLoss):
    assert loss_instr.lift(loss_module.loss) == loss_instr()
