# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway.schemas import Schema
from aioway.spaces import AnySpace, SchemaSpace


@pytest.fixture
def float_schema(fake_mode):
    return Schema.from_tensor(torch.randn(3))


@pytest.fixture
def int_schema(fake_mode):
    return Schema.from_tensor(torch.randn(3).to(torch.int))


@pytest.fixture(params=[float_schema.name, int_schema.name])
def any_schema(request: pytest.FixtureRequest):
    return request.getfixturevalue(request.param)


def test_any_space(any_schema: Schema):
    assert any_schema in AnySpace()


def test_schema_space(int_schema: Schema, float_schema: Schema):
    assert int_schema in SchemaSpace(int_schema)
    assert float_schema in SchemaSpace(float_schema)
    assert int_schema not in SchemaSpace(float_schema)
    assert float_schema not in SchemaSpace(int_schema)
