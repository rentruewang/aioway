# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch import nn

from aioway.spaces import unbounded_box_spec
from aioway.nets import emit_one
from aioway.trainers import NmfEmitter, NmfTrainer


@pytest.fixture
def nmf():
    with NmfEmitter(11).consider():
        yield


def test_emit_nmf(nmf):
    out = emit_one(unbounded_box_spec((3, 5)), unbounded_box_spec(()))

    assert isinstance(out, nn.Module)
    assert isinstance(out, NmfTrainer)
