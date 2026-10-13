# Copyright (c) AIoWay Authors - All Rights Reserved

"""The compat layer to pytree.

Everything runs on `torch.utils._cxx_pytree` (optree, C++), except key paths,
which `_cxx_pytree` does not implement and which therefore come from
`torch.utils._pytree`. Both walk a tree in the same leaf order, so paths from
one line up with leaves from the other.

Every spec this module returns is a `_cxx_pytree` spec, so specs from any
function here compare with each other. Specs made by code that calls
`torch.utils._pytree` directly are a different class and never compare equal.

Requires optree (`torch[optree]`).
"""

import functools
import sys
import typing
from collections import abc as cabc

from torch.utils import _cxx_pytree, _pytree

__all__ = [
    "Context",
    "GetAttrKey",
    "KeyEntry",
    "KeyPath",
    "LeafSpec",
    "MappingKey",
    "PyTree",
    "SequenceKey",
    "TreeSpec",
    "register_pytree_node",
    "tree_all",
    "tree_all_only",
    "tree_any",
    "tree_any_only",
    "tree_flatten",
    "tree_is_leaf",
    "tree_iter",
    "tree_leaves",
    "tree_map",
    "tree_map_",
    "tree_map_only",
    "tree_map_only_",
    "tree_structure",
    "tree_unflatten",
    "treespec_dumps",
    "treespec_loads",
    "treespec_pprint",
    "keystr",
    "tree_flatten_with_path",
    "tree_leaves_with_path",
    "tree_map_with_path",
]

# Types.
Context = _cxx_pytree.Context
PyTree = _cxx_pytree.PyTree
TreeSpec = _cxx_pytree.TreeSpec  # `optree.PyTreeSpec`, not `_pytree.TreeSpec`.
LeafSpec = _cxx_pytree.LeafSpec

# Key entries, for writing `flatten_with_keys_fn`. Only `_pytree` has them.
KeyEntry = _pytree.KeyEntry
KeyPath = _pytree.KeyPath
MappingKey = _pytree.MappingKey
SequenceKey = _pytree.SequenceKey
GetAttrKey = _pytree.GetAttrKey

# Registration.
# `_pytree`'s version, because it accepts `flatten_with_keys_fn` (the `_cxx_pytree`
# one raises on it) and also registers `flatten_fn` / `unflatten_fn` with
# `_cxx_pytree`, so one call covers both implementations. Note that `_cxx_pytree`
# passes `children` to `unflatten_fn` as a tuple, `_pytree` as a list.
# `flatten_with_keys_fn` must yield the same children in the same order as
# `flatten_fn`, or paths and leaves won't line up.
register_pytree_node = _pytree.register_pytree_node

# C++ implementations ====

tree_all = _cxx_pytree.tree_all
tree_all_only = _cxx_pytree.tree_all_only
tree_any = _cxx_pytree.tree_any
tree_any_only = _cxx_pytree.tree_any_only
tree_flatten = _cxx_pytree.tree_flatten
tree_is_leaf = _cxx_pytree.tree_is_leaf
tree_iter = _cxx_pytree.tree_iter
tree_leaves = _cxx_pytree.tree_leaves
tree_map = _cxx_pytree.tree_map
tree_map_ = _cxx_pytree.tree_map_
tree_map_only = _cxx_pytree.tree_map_only
tree_map_only_ = _cxx_pytree.tree_map_only_
tree_structure = _cxx_pytree.tree_structure
tree_unflatten = _cxx_pytree.tree_unflatten
treespec_dumps = _cxx_pytree.treespec_dumps
treespec_loads = _cxx_pytree.treespec_loads
treespec_pprint = _cxx_pytree.treespec_pprint

# Paths related ops are not supported by `_cxx_pytree`. ====

keystr = _pytree.keystr
tree_leaves_with_path = _pytree.tree_leaves_with_path
tree_map_with_path = _pytree.tree_map_with_path
key_get = _pytree.key_get


@functools.wraps(_pytree.tree_flatten_with_path)
def tree_flatten_with_path(
    tree: typing.Any, is_leaf: cabc.Callable[[typing.Any], bool] | None = None
) -> tuple[list[tuple[KeyPath, typing.Any]], TreeSpec]:
    # Since `_cxx_pytree` does not support this function,
    # but we want the `spec` to be compatible.
    flat = _pytree.tree_leaves_with_path(tree, is_leaf=is_leaf)
    spec = _cxx_pytree.tree_structure(tree, is_leaf=is_leaf)
    return flat, spec
