# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway.cases import NMFSpace, NMFTrainerModule, train_nmf
from aioway.emits import emit_one
from aioway.spaces import BoxSpace
from aioway.torch.nn import NnUFunc


@pytest.fixture
def nmf():
    with train_nmf.consider():
        yield


def test_emit_nmf(nmf):
    out = emit_one(
        NMFSpace(3, 5, 7),
        BoxSpace(torch.zeros([]), torch.ones([])),
    )

    assert isinstance(out, NnUFunc)
    assert isinstance(out.module, NMFTrainerModule)
