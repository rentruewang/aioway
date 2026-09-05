# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch import nn
from torchrl.data import tensor_specs as tspecs

from aioway.emits import emit_loss, emit_one, route_loss
from aioway.tspecs import ArgsTSpec, LossTSpec


@pytest.fixture
def emit_loss_context():
    with emit_loss.consider():
        yield


def test_route_unbounded_loss():
    found_losses = 0
    for _ in route_loss(tspecs.Unbounded(shape=5), tspecs.Unbounded(shape=5)):
        found_losses += 1
    assert found_losses


def test_route_to_losses(emit_loss_context):
    input_target_tspec = ArgsTSpec(
        input=tspecs.Unbounded(shape=5), target=tspecs.Unbounded(shape=5)
    )
    loss_tspec = LossTSpec()

    out = emit_one(input_target_tspec, loss_tspec)
    assert isinstance(out, nn.Module), out
    assert type(out).__name__.endswith("Loss"), out
