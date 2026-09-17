# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch import nn

from aioway.nn.rewrites import NetMorphLinearDeeper


def sequential():
    return nn.Sequential(
        nn.Linear(3, 5),
        nn.ReLU(),
        nn.Linear(5, 7),
        nn.ReLU6(),
        nn.Linear(7, 9),
    )


def _valid_sequential():
    seq = sequential()
    yield seq[:3]
    yield seq[2:5]


def _invalid_sequential():
    seq = sequential()
    yield seq[:2]
    yield seq[2:4]


@pytest.fixture(params=_valid_sequential())
def valid_sequential(request):
    return request.param


@pytest.fixture(params=_invalid_sequential())
def invalid_sequential(request):
    return request.param


@pytest.fixture
def deeper():
    return NetMorphLinearDeeper()


def test_valid_deeper_handle(
    deeper: NetMorphLinearDeeper, valid_sequential: nn.Sequential
):
    assert deeper.handle(valid_sequential)


def test_valid_deeper_handle(
    deeper: NetMorphLinearDeeper, valid_sequential: nn.Sequential
):
    out = deeper(valid_sequential)
    assert isinstance(out, nn.Sequential)


def test_invalid_deeper_no_handle(
    deeper: NetMorphLinearDeeper, invalid_sequential: nn.Sequential
):
    assert deeper(invalid_sequential) is NotImplemented
