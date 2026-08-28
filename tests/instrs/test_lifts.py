# Copyright (c) AIoWay Authors - All Rights Reserved


import pytest

from aioway.instrs import BaseLossInstr, Instr, list_lift_rules


@pytest.fixture(scope="module")
def registered_instr_type():
    return {rule.instr_type for rule in list_lift_rules()}


@pytest.fixture(params=BaseLossInstr.__subclasses__())
def loss_instr_cls(request: pytest.FixtureRequest):
    return request.param


def test_list_rules():
    assert list_lift_rules()


def test_loss_instr_cls(
    loss_instr_cls: type[Instr], registered_instr_type: set[type[Instr]]
):
    assert loss_instr_cls in registered_instr_type
