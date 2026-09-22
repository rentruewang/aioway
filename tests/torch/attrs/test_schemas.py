# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import tensordict as td
import torch

from aioway.torch import Attr, Schema


def make_attr(dtype: torch.dtype = torch.float32) -> Attr:
    return Attr.parse(torch.zeros(3, dtype=dtype))


# Most tests share this two-level schema: {"id": ID, "user": {"name": NAME}}
ID = make_attr()
NAME = make_attr()
USER = Schema({"name": NAME})
SCHEMA = Schema({"id": ID, "user": USER})


def test_len_counts_top_level_entries():
    assert len(SCHEMA) == 2
    assert len(Schema()) == 0


def test_getitem_by_name():
    assert SCHEMA["id"] is ID
    assert SCHEMA["user"] is USER


def test_getitem_by_path():
    # Same as SCHEMA[("user", "name")].
    assert SCHEMA["user", "name"] is NAME


def test_getitem_missing_raises_key_error():
    with pytest.raises(KeyError):
        SCHEMA["nope"]

    with pytest.raises(KeyError):
        SCHEMA["user", "nope"]


def test_get_returns_value_or_default():
    assert SCHEMA.get("id") is ID
    assert SCHEMA.get(("user", "name")) is NAME
    assert SCHEMA.get("nope") is None
    assert SCHEMA.get(("user", "nope"), "fallback") == "fallback"


def test_contains_by_name():
    assert "id" in SCHEMA
    assert "user" in SCHEMA
    assert "nope" not in SCHEMA


def test_contains_by_path():
    assert ("user", "name") in SCHEMA
    assert ("user", "nope") not in SCHEMA


def test_keys_are_top_level_by_default():
    assert list(SCHEMA.keys()) == ["id", "user"]


def test_keys_leaves_only():
    assert list(SCHEMA.keys(leaves_only=True)) == ["id"]


def test_keys_include_nested_gives_leaf_paths():
    assert list(SCHEMA.keys(include_nested=True)) == ["id", ("user", "name")]


def test_three_level_nesting():
    city = make_attr()
    schema = Schema({"user": Schema({"address": Schema({"city": city})})})

    assert schema["user", "address", "city"] is city
    all_nested_keys = {("user", "address", "city"), ("user", "address"), "user"}
    assert set(schema.keys(include_nested=True)) == all_nested_keys


def test_select_keeps_only_requested_entries():
    picked = SCHEMA.select("user")

    assert len(picked) == 1
    assert picked["user"] is USER


def test_select_ignores_missing_names_unless_strict():
    assert len(SCHEMA.select("id", "nope")) == 1

    with pytest.raises(KeyError):
        SCHEMA.select("id", "nope", strict=True)


def test_dtype_is_shared_dtype_or_none():
    assert SCHEMA.dtype == ID.dtype  # every leaf, nested included, is float32

    mixed = Schema({"a": make_attr(torch.float32), "b": make_attr(torch.int64)})
    assert mixed.dtype is None


def test_parse_builds_nested_schema():
    data = td.TensorDict(
        {"id": torch.zeros(3), "user": td.TensorDict({"name": torch.zeros(3)})},
        batch_size=[3],
    )
    schema = Schema.parse(data)

    assert isinstance(schema["id"], Attr)
    assert isinstance(schema["user"], Schema)
    assert isinstance(schema["user", "name"], Attr)
