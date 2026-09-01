# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torchrl.data import tensor_specs as tspecs

from aioway.tspecs import is_tspec_like


def _tspecs():
    yield tspecs.Unbounded(3)
    yield tspecs.Bounded(4, 5, 6)
    yield tspecs.BoundedContinuous(5, 6, 7)
    yield tspecs.BoundedDiscrete(6, 7, 8)


@pytest.fixture(params=_tspecs())
def tspec(request):
    return request.param


class _C:
    pass


def _not_tspec():
    yield object()
    yield _C()


@pytest.fixture(params=_not_tspec())
def not_tspec(request):
    return request.param


def test_tspec_protocol(tspec: tspecs.TensorSpec):
    """
    Test if the `TSpec` protocol captures the types given directly by `torchrl`.
    """

    assert is_tspec_like(tspec)


def test_tspec_type(tspec):
    assert is_tspec_like(tspec)


def test_not_tspec_type(not_tspec):
    assert not is_tspec_like(not_tspec)
