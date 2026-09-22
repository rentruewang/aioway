# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import tensordict as td
import torch

from aioway.torch import Attr, Schema


def make_attr(dtype: torch.dtype = torch.float32) -> Attr:
    return Attr.parse(torch.zeros(3, dtype=dtype))


@pytest.fixture
def attr() -> Attr:
    return make_attr()


@pytest.fixture
def user(attr) -> Schema:
    return Schema({"name": attr})


@pytest.fixture
def schema(attr, user) -> Schema:
    return Schema({"id": attr, "user": user})


def test_length(schema):
    assert len(schema) == 2
    assert len(Schema()) == 0


def test_getitem_shallow(schema, attr, user):
    assert schema["id"] is attr
    assert schema["user"] is user


def test_getitem_nested(schema, attr):
    # Same as schema[("user", "name")].
    assert schema["user", "name"] is attr


def test_getitem_missing(schema):
    with pytest.raises(KeyError):
        schema["nope"]

    with pytest.raises(KeyError):
        schema["user", "nope"]


def test_get_with_item(schema, attr):
    assert schema.get("id") is attr
    assert schema.get(("user", "name")) is attr
    assert schema.get("nope") is None
    assert schema.get(("user", "nope"), "fallback") == "fallback"


def test_contains_by_name(schema):
    assert "id" in schema
    assert "user" in schema
    assert "nope" not in schema


def test_contains_nested(schema):
    assert ("user", "name") in schema
    assert ("user", "nope") not in schema


def test_keys_default(schema):
    assert list(schema.keys()) == ["id", "user"]


def test_keys_leaves_only(schema):
    assert list(schema.keys(leaves_only=True)) == ["id"]


def test_keys_includes_nested(schema):
    assert list(schema.keys(include_nested=True)) == ["id", ("user", "name"), "user"]


def test_more_nesting():
    city = make_attr()
    deep = Schema({"user": Schema({"address": Schema({"city": city})})})

    assert deep["user", "address", "city"] is city
    all_nested_keys = {("user", "address", "city"), ("user", "address"), "user"}
    assert set(deep.keys(include_nested=True)) == all_nested_keys


def test_select(schema, user):
    picked = schema.select("user")

    assert len(picked) == 1
    assert picked["user"] is user


def test_select_ignores_missing(schema):
    assert len(schema.select("id", "nope")) == 1

    with pytest.raises(KeyError):
        schema.select("id", "nope", strict=True)


def test_dtype_coerce(schema, attr):
    assert schema.dtype == attr.dtype  # every leaf, nested included, is float32

    mixed = Schema({"a": make_attr(torch.float32), "b": make_attr(torch.int64)})
    assert mixed.dtype is None


def test_parse_nested():
    data = td.TensorDict(
        {"id": torch.zeros(3), "user": td.TensorDict({"name": torch.zeros(3)})},
        batch_size=[3],
    )
    parsed = Schema.parse(data)

    assert isinstance(parsed["id"], Attr)
    assert isinstance(parsed["user"], Schema)
    assert isinstance(parsed["user", "name"], Attr)
