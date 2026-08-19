# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch import nn

from aioway._tspecs import unbounded_box_space
from aioway.nets import emit_one
from aioway.trainers import NmfEmitter, NmfTrainer


@pytest.fixture
def nmf():
    with NmfEmitter(11).consider():
        yield


def test_emit_nmf(nmf):
    out = emit_one(unbounded_box_space((3, 5)), unbounded_box_space(()))

    assert isinstance(out, nn.Module)
    assert isinstance(out, NmfTrainer)
