# Copyright (c) AIoWay Authors - All Rights Reserved

"Tests for `aioway.t` and the pytree mixins."

import dataclasses as dcls

import pytest
import torch
from torch.utils import _pytree

from aioway.t import (
    SUPPORTED_NODES,
    MappingKey,
    PyTreeDataclassMixin,
    PyTreeDictMixin,
    TreeSpec,
    key_get,
    keystr,
    register_pytree_node,
    tree_flatten,
    tree_flatten_with_path,
    tree_leaves,
    tree_map,
    tree_structure,
    tree_unflatten,
)


@dcls.dataclass
class Linear(PyTreeDataclassMixin):
    weight: torch.Tensor
    bias: torch.Tensor | None = None
    numel: int = dcls.field(init=False)

    def __post_init__(self) -> None:
        self.numel = self.weight.numel()


@dcls.dataclass(frozen=True, slots=True)
class Model(PyTreeDataclassMixin):
    layers: list[Linear]
    head: Linear


class Bag(PyTreeDictMixin):
    "Keys can differ between instances."

    _PY_TREE_KEY = MappingKey

    def __init__(self, **items) -> None:
        self.items = items

    def _tree_dict(self):
        return dict(self.items)

    @property
    def _tree_context(self):
        return None

    @classmethod
    def _tree_from_dict(cls, mapping, context):
        return cls(**mapping)


class SubBag(Bag):
    "Inherits `_PY_TREE_KEY`."


class KeylessBag(Bag):
    _PY_TREE_KEY = NotImplemented


def model() -> Model:
    return Model(
        layers=[Linear(torch.zeros(2)), Linear(torch.ones(3), torch.ones(1))],
        head=Linear(torch.ones(4)),
    )


class TestReg:

    def test_registered(self):
        m = model()
        assert len(tree_leaves(m)) == len(_pytree.tree_leaves(m)) == 6

    def test_supported_nodes(self):
        assert SUPPORTED_NODES is _pytree.SUPPORTED_NODES
        assert {Linear, Model, Bag, SubBag} <= SUPPORTED_NODES.keys()

    def test_reg_twice_fail(self):
        with pytest.raises(ValueError):
            register_pytree_node(Bag, Bag._tree_flatten, Bag._tree_unflatten)


class TestDcls:

    def test_round_trip(self):
        m = model()
        leaves, spec = tree_flatten(m)
        out = tree_unflatten([x if x is None else x + 1 for x in leaves], spec)

        assert type(out) is Model and type(out.head) is Linear
        assert torch.equal(out.head.weight, m.head.weight + 1)
        assert out.layers[0].bias is None
        assert out.head.numel == 4  # `init=False` field recomputed.

    def test_paths(self):
        flat, _ = tree_flatten_with_path(Linear(torch.zeros(1)))
        assert [keystr(path) for path, _ in flat] == [".weight", ".bias"]

    def test_same_different_data(self):
        with_none = tree_structure(Linear(torch.zeros(1)))
        with_bias = tree_structure(Linear(torch.zeros(1), torch.zeros(1)))
        assert with_none == with_bias

    def test_not_dcls_err(self):
        class NotADataclass(PyTreeDataclassMixin):
            pass

        with pytest.raises(TypeError):
            tree_leaves(NotADataclass())


class TestTreeSpec:

    def test_eq(self):
        assert tree_structure(model()) == tree_structure(model())

    def test_ne_field(self):
        assert tree_structure(Bag(a=1)) != tree_structure(Bag(b=1))

    def test_ne_type(self):
        assert tree_structure(Bag(a=1)) != tree_structure(SubBag(a=1))


def test_tree_map_same_key():
    with pytest.raises(ValueError):
        tree_map(lambda x, y: x + y, Bag(a=1), Bag(b=2))


def test_flatten_with_path():
    m = model()
    flat, spec = tree_flatten_with_path(m)

    assert isinstance(spec, TreeSpec)
    assert spec == tree_structure(m)
    assert all(v is leaf for (_, v), leaf in zip(flat, tree_leaves(m), strict=True))
    assert all(key_get(m, path) is v for path, v in flat)


def test_subclass_key():
    flat, _ = tree_flatten_with_path(SubBag(x=1))
    assert [keystr(path) for path, _ in flat] == ["['x']"]


def test_missing_key_no_paths():
    assert tree_leaves(KeylessBag(x=1)) == [1]

    with pytest.raises(TypeError):
        tree_flatten_with_path(KeylessBag(x=1))


def test_tensordict_round_trip():
    # Use `importskip` because importing causes type check issues here.
    td = pytest.importorskip("tensordict")

    tdict = td.TensorDict(
        {"a": torch.zeros(2), "n": {"b": torch.ones(2)}}, batch_size=[2]
    )

    leaves, spec = tree_flatten({"td": tdict})
    out = tree_unflatten([x + 1 for x in leaves], spec)["td"]

    assert isinstance(out, td.TensorDict)
    assert out.batch_size == tdict.batch_size
    assert torch.equal(out["n", "b"], tdict["n", "b"] + 1)
