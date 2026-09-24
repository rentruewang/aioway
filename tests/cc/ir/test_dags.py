# Copyright (c) AIoWay Authors - All Rights Reserved

from collections import abc as cabc

import pytest
import torch

from aioway.cc import Dag, DoneThunk
from aioway.torch import fake_mode


def add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left + right


def scale(tensor: torch.Tensor, factor: float) -> torch.Tensor:
    return tensor * factor


@pytest.fixture
def fakes() -> tuple[torch.Tensor, torch.Tensor]:
    with fake_mode():
        return torch.zeros(3), torch.zeros(3)


@pytest.fixture
def thunks(fakes) -> tuple[DoneThunk, ...]:
    x0, x1 = fakes

    with fake_mode():
        y = add(x0, x1)
        z = scale(y, factor=2)

    return (
        DoneThunk(func=add, args=(x0, x1), kwargs={}, result=y),
        DoneThunk(func=scale, args=(y,), kwargs={"factor": 2}, result=z),
    )


@pytest.fixture
def dag(thunks) -> Dag:
    return Dag(thunks)


def test_dag_len_is_thunks(dag):
    assert len(dag) == 2


def test_dag_not_empty():
    with pytest.raises(ValueError):
        Dag([])


def test_inputs_are_fakes(dag: Dag, fakes: cabc.Sequence[torch.Tensor]):
    assert len(dag.inputs()) == 2
    assert all(got is want for got, want in zip(dag.inputs(), fakes))


def test_call_runs(dag: Dag):
    real = dag(torch.ones(3), torch.full((3,), 2.0))

    assert torch.equal(real, torch.full((3,), 6.0))


def test_call_twice(dag: Dag):
    first = dag(torch.ones(3), torch.ones(3))
    second = dag(torch.ones(3), torch.ones(3))

    assert torch.equal(first, second)


def test_wrong_number_of_inputs(dag: Dag):
    with pytest.raises(TypeError):
        dag(torch.ones(3))


def test_output_unique(thunks):
    with pytest.raises(KeyError):
        Dag([thunks[0], thunks[0]])
