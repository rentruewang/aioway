# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway.cc import InstrSet, Exec


@pytest.fixture
def exec(dag: InstrSet) -> Exec:
    return Exec(dag)


def test_len(exec: Exec) -> None:
    assert len(exec) == 2


def test_getitem(exec: Exec, thunks):
    assert exec[0] is thunks[0]


def test_iter(exec: Exec, thunks):
    assert list(exec) == list(thunks)


def test_inputs_are_fakes(exec: Exec, fakes):
    assert set(exec.inputs) == set(fakes)


def test_call_runs(exec: Exec):
    real = exec(torch.ones(3), torch.full((3,), 2.0))

    assert torch.equal(real, torch.full((3,), 6.0))


def test_call_twice(exec: Exec):
    first = exec(torch.ones(3), torch.ones(3))
    second = exec(torch.ones(3), torch.ones(3))

    assert torch.equal(first, second)


def test_wrong_input_signature(exec: Exec):
    with pytest.raises(TypeError):
        exec(torch.ones(3))
