# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torchrl.data import tensor_specs as tspecs

from aioway.tspecs import TSpec, is_tspec_type


def _tspec_types():
    yield TSpec
    yield tspecs.Unbounded
    yield tspecs.Bounded
    yield tspecs.BoundedContinuous
    yield tspecs.BoundedDiscrete


@pytest.fixture(params=_tspec_types())
def tspec_type(request):
    return request.param


class _C:
    pass


def _not_tspec_types():
    yield object
    yield _C


@pytest.fixture(params=_not_tspec_types())
def not_tspec_type(request):
    return request.param


def test_tspec_type(tspec_type):
    assert is_tspec_type(tspec_type)


def test_not_tspec_type(not_tspec_type):
    assert not is_tspec_type(not_tspec_type)
