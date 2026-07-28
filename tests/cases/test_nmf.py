# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn

from aioway.cases import NMFSpace, NMFTrainerModule, train_nmf
from aioway.emits import emit_one
from aioway.spaces import BoxSpace


@pytest.fixture
def nmf():
    with train_nmf.consider():
        yield


def test_emit_nmf(nmf):
    out = emit_one(
        NMFSpace(3, 5, 7),
        BoxSpace(torch.zeros([]), torch.ones([])),
    )

    assert isinstance(out, nn.Module)
    assert isinstance(out, NMFTrainerModule)
