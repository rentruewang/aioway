# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
import torch

from aioway.compilers import NormEmitter, NormType, emit_one
from aioway.tspecs import unbounded_box_tspec


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
def spec(ndim, num_features):
    shape = list(range(5, 5 + ndim))
    return unbounded_box_tspec(shape=[num_features, *shape])


def test_batch_inst_norm(batch_inst_norm_emitter, spec):
    mod = emit_one(spec, spec)
    input = spec.sample(torch.Size([13]))
    output = mod.module()(input)
    assert output.shape == input.shape
