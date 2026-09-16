# Copyright (c) AIoWay Authors - All Rights Reserved

import torch
import pytest
from aioway.tensors import batch_tspec, iter_tspec
from torchrl.data import tensor_specs as tspecs


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
