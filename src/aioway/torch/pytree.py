# Copyright (c) AIoWay Authors - All Rights Reserved

"Extension of `pytree` from `torch`."

import typing
from collections import abc as cabc

import torch
from torch.utils import _pytree as pytree

from aioway._utils import AnyDict

__all__ = [
    "tree_leaves_typed",
    "tree_map_memo",
    "find_nested_tensors",
    "replace_tensors",
]


def tree_map_memo(
    obj,
    replace: cabc.Callable[..., object],
    memo: AnyDict[typing.Any, typing.Any] | None = None,
) -> typing.Any:
    """
    Decompose and replace. When this is called, `replace(obj)` is directly invoked.
    If it returns `NotImplemented`, then decomposing would continue.

    Args:
        obj: The object to maybe replace.
        types: The types to replace.
        replace: The replacer function.
        memo:
            Like `memo` in `__deepcopy__`,
            this is s.t. don't replace the same item with different ones.
    """

    memo = memo or AnyDict()

    def replace_cached(item):
        if item not in memo:
            # If it is `NotImplemented`, no replacement is made.
            if (result := replace(item)) is NotImplemented:
                result = item

            memo[item] = result

        return memo[item]

    return pytree.tree_map(replace_cached, obj)


def tree_leaves_typed(obj, *types: type) -> cabc.Iterator[typing.Any]:
    "Decompose the object based on the desired type."

    found = lambda item: isinstance(item, types)

    for elem in pytree.tree_leaves(obj, is_leaf=found):
        if found(elem):
            yield elem


def find_nested_tensors(
    obj: object, *, only_tensors: bool = False
) -> cabc.Iterator[torch.Tensor]:
    """
    Find and unpack tensors from containers.

    If `only_tensors` is `True`, raies an error
    if `obj` cannot be decomposed into purely tensors.
    """

    for item in pytree.tree_leaves(obj):
        if isinstance(item, torch.Tensor):
            yield item


def replace_tensors(
    obj: object, replace: cabc.Callable[[torch.Tensor], object]
) -> object:
    """
    Replace tensors whenever encountered with the given function.

    This function has the `__torch_function__` disabled in the scope of the rendering,
    because it can mess with attribute access, which oftentimes means that
    this function fails also during debugging if `__torch_function__` is not disabled.
    Caused by `.device` / `.shape` / `.dtype` calls, which is used in `replace_tensors`.
    """

    from .overrides import mode_off

    def maybe_replace(item):
        if not isinstance(item, torch.Tensor):
            return NotImplemented

        return replace(item)

    with mode_off():
        return tree_map_memo(obj, maybe_replace)
