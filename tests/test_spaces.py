# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway.schemas import Schema
from aioway.spaces import AnySpace, ContinuousSpace, DiscreteSpace


@pytest.fixture
def continuous_schema(fake_mode):
    return Schema.from_tensor(torch.randn(3))


@pytest.fixture
def discrete_schema(fake_mode):
    return Schema.from_tensor(torch.randn(3).int())


@pytest.fixture(params=[continuous_schema.name, discrete_schema.name])
def any_schema(request: pytest.FixtureRequest):
    return request.getfixturevalue(request.param)


def test_any_space(any_schema: Schema):
    assert any_schema in AnySpace()


def test_discrete_schema_space(discrete_schema: Schema):
    assert discrete_schema in DiscreteSpace(1)
    assert discrete_schema not in DiscreteSpace(2)
    assert discrete_schema not in DiscreteSpace(0)


def test_discrete_schema_continuous_space(discrete_schema: Schema):
    assert discrete_schema not in ContinuousSpace(1)
    assert discrete_schema not in ContinuousSpace(2)
    assert discrete_schema not in ContinuousSpace(0)


def test_continuous_schema_space(continuous_schema: Schema):
    assert continuous_schema in ContinuousSpace(1)
    assert continuous_schema not in ContinuousSpace(2)
    assert continuous_schema not in ContinuousSpace(0)


def test_continuous_schema_discrete_space(continuous_schema: Schema):
    assert continuous_schema not in DiscreteSpace(1)
    assert continuous_schema not in DiscreteSpace(2)
    assert continuous_schema not in DiscreteSpace(0)
