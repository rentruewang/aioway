# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway._utils import any_dict, any_set


def test_any_set_empty():
    s = any_set()
    assert len(s) == 0
    assert not s


def test_any_set_with_defaults():
    s = any_set(int, 1, 2, 3)
    assert len(s) == 3
    assert 1 in s
    assert 4 not in s
    assert sorted(s) == [1, 2, 3]


def test_any_set_dedupes():
    s = any_set(int, 1, 1, 2)
    assert len(s) == 2


def test_any_set_unhashable_uses_identity():
    a, b = [1], [1]
    s = any_set(list, a)
    assert a in s
    assert b not in s  # equal, but a different object


def test_any_set_add_and_discard():
    s = any_set(int)
    s.add(1)
    assert 1 in s
    s.discard(1)
    assert 1 not in s
    s.discard(1)  # no error when missing


def test_any_set_wrong_type_raises():
    s = any_set(int, 1)
    with pytest.raises(TypeError):
        "a" in s


def test_any_set_sub():
    a = any_set(int, 1, 2, 3)
    b = any_set(int, 2)
    assert sorted(a - b) == [1, 3]
    assert sorted(a) == [1, 2, 3]  # original untouched


def test_any_set_isub():
    a = any_set(int, 1, 2, 3)
    a -= any_set(int, 2)
    assert sorted(a) == [1, 3]


def test_any_set_isdisjoint():
    s = any_set(int, 1, 2)
    assert s.isdisjoint([3, 4])
    assert not s.isdisjoint([2, 3])


def test_any_dict_empty():
    d = any_dict()
    assert len(d) == 0
    assert not d


def test_obj_isdisjoint():
    a = object()
    b = object()
    c = object()

    s = any_set(object, a, b)

    assert s.isdisjoint([c])
    assert not s.isdisjoint([a])
    assert not s.isdisjoint([b])
    assert not s.isdisjoint([a, c])


def test_any_dict_with_defaults():
    d = any_dict(str, ("a", 1), ("b", 2))
    assert d["a"] == 1
    assert d["b"] == 2
    assert len(d) == 2


def test_any_dict_set_get_del():
    d = any_dict(str)
    d["a"] = 1
    assert d["a"] == 1
    d["a"] = 2
    assert d["a"] == 2
    del d["a"]
    assert "a" not in d


def test_any_dict_missing_key_raises():
    d = any_dict(str)
    with pytest.raises(KeyError):
        d["a"]
    with pytest.raises(KeyError):
        del d["a"]


def test_any_dict_unhashable_keys():
    k1, k2 = [1], [1]
    d = any_dict(list, (k1, "x"))
    assert d[k1] == "x"
    assert k2 not in d


def test_any_dict_get():
    d = any_dict(str, ("a", 1))
    assert d.get("a") == 1
    assert d.get("b") is None
    assert d.get("b", 0) == 0


def test_any_dict_keys_values_items():
    d = any_dict(str, ("a", 1), ("b", 2))
    assert sorted(d.keys()) == ["a", "b"]
    assert sorted(d.values()) == [1, 2]
    assert sorted(d.items()) == [("a", 1), ("b", 2)]


def test_any_dict_or():
    a = any_dict(str, ("a", 1))
    b = any_dict(str, ("b", 2))
    c = a | b
    assert sorted(c.items()) == [("a", 1), ("b", 2)]
    assert sorted(a.items()) == [("a", 1)]  # original untouched


def test_any_dict_ior():
    a = any_dict(str, ("a", 1))
    a |= any_dict(str, ("a", 10), ("b", 2))
    assert sorted(a.items()) == [("a", 10), ("b", 2)]
