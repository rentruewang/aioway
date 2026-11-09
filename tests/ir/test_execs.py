# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway.ir import IrExec, Proc


@pytest.fixture
def exec(dag: Proc) -> IrExec:
    return IrExec(dag)


def test_len(exec: IrExec) -> None:
    assert len(exec) == 2


def test_getitem(exec: IrExec, thunks):
    assert exec[0] is thunks[0]


def test_iter(exec: IrExec, thunks):
    assert list(exec) == list(thunks)


def test_inputs_are_fakes(exec: IrExec, fakes):
    assert set(exec.proc.inputs.values()) == set(fakes)


def test_call_runs(exec: IrExec):
    real = exec(torch.ones(3), torch.full((3,), 2.0))

    assert torch.equal(real, torch.full((3,), 6.0))


def test_call_twice(exec: IrExec):
    first = exec(torch.ones(3), torch.ones(3))
    second = exec(torch.ones(3), torch.ones(3))

    assert torch.equal(first, second)


def test_wrong_input_signature(exec: IrExec):
    with pytest.raises(TypeError):
        exec(torch.ones(3))
