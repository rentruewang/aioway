# Copyright (c) AIoWay Authors - All Rights Reserved

"The compat layer to pytree (first class supported)."

# Import as private s.t. we do not need to define an `__all__`.

import functools as _functools
import typing as _typing
from collections import abc as _cabc

from torch.utils import _cxx_pytree, _pytree

# Types.
Context = _cxx_pytree.Context
PyTree = _cxx_pytree.PyTree
TreeSpec = _cxx_pytree.TreeSpec
LeafSpec = _cxx_pytree.LeafSpec

# Registration. Should work everywhere. ====
SUPPORTED_NODES = _pytree.SUPPORTED_NODES
SUPPORTED_SERIALIZED_TYPES = _pytree.SUPPORTED_SERIALIZED_TYPES
SERIALIZED_TYPE_TO_PYTHON_TYPE = _pytree.SERIALIZED_TYPE_TO_PYTHON_TYPE
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

# Namedtuple / structseq checks. ====

is_namedtuple = _cxx_pytree.is_namedtuple
is_namedtuple_class = _cxx_pytree.is_namedtuple_class
is_namedtuple_instance = _cxx_pytree.is_namedtuple_instance
is_structseq = _cxx_pytree.is_structseq
is_structseq_class = _cxx_pytree.is_structseq_class
is_structseq_instance = _cxx_pytree.is_structseq_instance

# Paths related ops are not supported by `_cxx_pytree`. ====

KeyEntry = _pytree.KeyEntry
KeyPath = _pytree.KeyPath
MappingKey = _pytree.MappingKey
SequenceKey = _pytree.SequenceKey
GetAttrKey = _pytree.GetAttrKey


keystr = _pytree.keystr
tree_leaves_with_path = _pytree.tree_leaves_with_path
tree_map_with_path = _pytree.tree_map_with_path
key_get = _pytree.key_get


@_functools.wraps(_pytree.tree_flatten_with_path)
def tree_flatten_with_path(
    tree: _typing.Any, is_leaf: _cabc.Callable[[_typing.Any], bool] | None = None
) -> tuple[list[tuple[KeyPath, _typing.Any]], TreeSpec]:
    # Since `_cxx_pytree` does not support this function,
    # but we want the `spec` to be compatible.
    flat = _pytree.tree_leaves_with_path(tree, is_leaf=is_leaf)
    spec = _cxx_pytree.tree_structure(tree, is_leaf=is_leaf)
    return flat, spec
