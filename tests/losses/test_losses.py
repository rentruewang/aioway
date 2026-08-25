# Copyright (c) AIoWay Authors - All Rights Reserved

from torchrl.data import tensor_specs as tspecs

from aioway.losses import route_loss


def test_route_unbounded_loss():
    found_losses = 0
    for _ in route_loss(tspecs.Unbounded(shape=5), tspecs.Unbounded(shape=5)):
        found_losses += 1
    assert found_losses
