# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest

from aioway.spaces import unbounded_box_spec
from aioway.nets import NormEmitter, NormType, emit_one, layer_norm_emitter


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
def batch_inst_norm_emitter(norm):
    with NormEmitter(norm).consider():
        yield


@pytest.fixture
def layer_norm():
    with layer_norm_emitter.consider():
        yield


@pytest.fixture
def spec(ndim, num_features):
    shape = list(range(5, 5 + ndim))
    return unbounded_box_spec(shape=[num_features, *shape])


def test_batch_inst_norm(batch_inst_norm_emitter, spec):
    mod = emit_one(spec, spec)
    input = spec.sample([13])
    output = mod(input)
    assert output.shape == input.shape


def test_layer_norm(layer_norm, spec):
    mod = emit_one(spec, spec)
    input = spec.sample([13])
    output = mod(input)
    assert output.shape == input.shape
