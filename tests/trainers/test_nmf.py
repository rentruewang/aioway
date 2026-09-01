# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch import nn

from aioway.compilers import emit_one
from aioway.trainers import NmfEmitter, NmfTrainer
from aioway.tspecs import unbounded_box_tspec


@pytest.fixture
def nmf():
    with NmfEmitter(11).consider():
        yield


def test_emit_nmf(nmf):
    out = emit_one(unbounded_box_tspec((3, 5)), unbounded_box_tspec(()))

    assert isinstance(out, nn.Module)
    assert isinstance(out, NmfTrainer)
