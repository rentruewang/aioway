# Copyright (c) AIoWay Authors - All Rights Reserved

from torchrl import data as rldata

from aioway.losses import route_loss


def test_route_unbounded_loss():
    found_losses = 0
    for _ in route_loss(rldata.Unbounded(shape=5), rldata.Unbounded(shape=5)):
        found_losses += 1
    assert found_losses
