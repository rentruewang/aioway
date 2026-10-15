# Copyright (c) AIoWay Authors - All Rights Reserved


import pytest

from aioway._utils import any_dict, any_set


class TestAnySet:
    def test_empty(self):
        s = any_set()
        assert len(s) == 0
        assert not s

    def test_with_defaults(self):
        s = any_set(1, 2, 3)
        assert len(s) == 3
        assert 1 in s
        assert 4 not in s
        assert sorted(s) == [1, 2, 3]

    def test_dedupes(self):
        s = any_set(1, 1, 2)
        assert len(s) == 2

    def test_unhashable_uses_identity(self):
        a, b = [1], [1]
        s = any_set(a)
        assert a in s
        assert b not in s  # equal, but a different object

    def test_add_and_discard(self):
        s = any_set()
        s.add(1)
        assert 1 in s
        s.discard(1)
        assert 1 not in s
        s.discard(1)  # no error when missing

    def test_eq(self):
        s = any_set(1, 2)
        assert s == [2, 1]
        assert s != [1]
        assert s != [1, 2, 3]

    def test_eq_unhashable(self):
        a = [1]
        s = any_set(a)
        assert s == [a]
        assert s != [[1]]  # equal, but a different object

    def test_sub(self):
        a = any_set(1, 2, 3)
        b = any_set(2)
        assert sorted(a - b) == [1, 3]
        assert sorted(a) == [1, 2, 3]  # original untouched

    def test_isub(self):
        a = any_set(1, 2, 3)
        a -= any_set(2)
        assert sorted(a) == [1, 3]

    def test_isub_self(self):
        s = any_set(1, 2)
        s -= s
        assert len(s) == 0

    def test_and(self):
        s = any_set(1, 2, 3) & any_set(2, 3, 4)
        assert sorted(s) == [2, 3]

    def test_isdisjoint(self):
        s = any_set(1, 2)
        assert s.isdisjoint([3, 4])
        assert not s.isdisjoint([2, 3])

    def test_isdisjoint_object(self):
        a = object()
        b = object()
        c = object()

        s = any_set(a, b)

        assert s.isdisjoint([c])
        assert not s.isdisjoint([a])
        assert not s.isdisjoint([b])
        assert not s.isdisjoint([a, c])

    def test_isdisjoint_torch(self):
        import torch

        a = torch.tensor([1, 2])
        b = torch.tensor([1, 2])

        s = any_set(a)

        # Same object -> not disjoint.
        assert not s.isdisjoint([a])

        # Different tensor, despite == producing an elementwise tensor -> disjoint.
        assert s.isdisjoint([b])

    def test_isdisjoint_no_mutation(self):
        a = object()
        b = object()

        s = any_set(a)

        assert s.isdisjoint([b])
        assert len(s) == 1
        assert a in s

    def test_id_is_not_hash(self):
        u = [1]
        s = any_set()
        s.add(u)
        assert id(u) not in s
        s.add(id(u))
        assert len(s) == 2

    def test_hash_per_object(self):
        any_set((1, [2]))
        s = any_set((1, 2))
        assert tuple([1, 2]) in s


class TestAnyDict:
    def test_empty(self):
        d = any_dict()
        assert len(d) == 0
        assert not d

    def test_with_defaults(self):
        d = any_dict(str, ("a", 1), ("b", 2))
        assert d["a"] == 1
        assert d["b"] == 2
        assert len(d) == 2

    def test_set_get_del(self):
        d = any_dict(str)
        d["a"] = 1
        assert d["a"] == 1
        d["a"] = 2
        assert d["a"] == 2
        del d["a"]
        assert "a" not in d

    def test_missing_key_raises(self):
        d = any_dict(str)
        with pytest.raises(KeyError):
            d["a"]
        with pytest.raises(KeyError):
            del d["a"]

    def test_unhashable_keys(self):
        k1, k2 = [1], [1]
        d = any_dict(list, (k1, "x"))
        assert d[k1] == "x"
        assert k2 not in d

    def test_get(self):
        d = any_dict(str, ("a", 1))
        assert d.get("a") == 1
        assert d.get("b") is None
        assert d.get("b", 0) == 0

    def test_keys_values_items(self):
        d = any_dict(str, ("a", 1), ("b", 2))
        assert sorted(d.keys()) == ["a", "b"]
        assert sorted(d.values()) == [1, 2]
        assert sorted(d.items()) == [("a", 1), ("b", 2)]

    def test_eq(self):
        a = any_dict(str, ("a", 1), ("b", 2))
        assert a == any_dict(str, ("b", 2), ("a", 1))
        assert a == {"a": 1, "b": 2}
        assert a != {"a": 1, "b": 3}
        assert a != {"a": 1}

    def test_or(self):
        a = any_dict(str, ("a", 1))
        b = any_dict(str, ("b", 2))
        c = a | b
        assert sorted(c.items()) == [("a", 1), ("b", 2)]
        assert sorted(a.items()) == [("a", 1)]  # original untouched

    def test_ior(self):
        a = any_dict(str, ("a", 1))
        a |= any_dict(str, ("a", 10), ("b", 2))
        assert sorted(a.items()) == [("a", 10), ("b", 2)]

    def test_copy_copies(self):
        a = any_dict(str, ("a", 1))
        b = a.copy()
        b["b"] = 2
        del b["a"]
        assert sorted(a.items()) == [("a", 1)]

    def test_no_set_ops(self):
        d = any_dict(str, ("a", 1))
        assert not hasattr(d, "add")
        assert not hasattr(d, "discard")
        assert not hasattr(d, "__sub__")
