# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from numpy import random as np_rand
from torchrl.data import tensor_specs as tspecs

from aioway.torch import batch_tspec, iter_tspec


@pytest.fixture
def shuffled_ints() -> list[int]:
    ints = np_rand.randint(2, 10, [5])
    return [int(i) for i in ints]


@pytest.fixture
def ub_disc():
    return tspecs.UnboundedDiscrete()


@pytest.fixture
def ub_cont():
    return tspecs.UnboundedContinuous()


@pytest.fixture(params=[ub_disc.name, ub_cont.name])
def unbounded(request: pytest.FixtureRequest) -> tspecs.Unbounded:
    return request.getfixturevalue(request.param)


def test_batch_tspec(unbounded: tspecs.Unbounded, fake_mode):
    sampled = unbounded.sample(torch.Size([10]))

    assert batch_tspec(sampled) == unbounded


def test_iter_tspec(unbounded: tspecs.Unbounded, shuffled_ints: list[int], fake_mode):
    sampled = [unbounded.sample(torch.Size([i])) for i in shuffled_ints]
    assert iter_tspec(sampled) == unbounded
