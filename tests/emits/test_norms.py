# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway._specs import unbounded_box_spec
import typing
import pytest
from aioway.emits import NormEmitter, NormType, emit_one


@pytest.fixture(params=typing.get_args(NormType.__value__))
def norm(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def ndim(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=[11, 13, 17])
def num_features(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture
def norm_emitter(norm):
    with NormEmitter(norm).consider():
        yield


@pytest.fixture
def spec(ndim, num_features):
    shape = list(range(5, 5 + ndim))
    return unbounded_box_spec(shape=[num_features, *shape])


def test_emit_norm(norm_emitter, spec):
    mod = emit_one(spec, spec)
    input = spec.sample([13])
    output = mod(input)
    assert output.shape == input.shape
