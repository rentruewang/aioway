# Copyright (c) AIoWay Authors - All Rights Reserved

"Utility functions."

import typing
from collections import abc as cabc

import torch

from aioway._utils import AnyDict, any_dict

from .compat import tree_all, tree_leaves, tree_map

__all__ = [
    "tree_leaves_typed",
    "tree_map_memo",
    "find_nested_tensors",
    "replace_tensors",
    "is_tensor_container",
]


def tree_map_memo(
    obj, replace: cabc.Callable[..., object], memo: AnyDict | None = None
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

    memo = memo if memo is not None else any_dict()

    def replace_cached(item):
        if item not in memo:
            # If it is `NotImplemented`, no replacement is made.
            if (result := replace(item)) is NotImplemented:
                result = item

            memo[item] = result

        return memo[item]

    return tree_map(replace_cached, obj)


def tree_leaves_typed(obj, *types: type) -> list[typing.Any]:
    "Decompose the object based on the desired type."

    found = lambda item: isinstance(item, types)

    # Tree leaves without decomposing the target types.
    leaves = tree_leaves(obj, is_leaf=found)
    return [elem for elem in leaves if found(elem)]


def find_nested_tensors(obj: object) -> cabc.Iterator[torch.Tensor]:
    """
    Find and unpack tensors from containers.
    """

    yield from tree_leaves_typed(obj, torch.Tensor)


def replace_tensors(
    obj: object, replace: cabc.Callable[[torch.Tensor], object]
) -> typing.Any:
    """
    Replace tensors whenever encountered with the given function.

    This function has the `__torch_function__` disabled in the scope of the rendering,
    because it can mess with attribute access, which oftentimes means that
    this function fails also during debugging if `__torch_function__` is not disabled.
    Caused by `.device` / `.shape` / `.dtype` calls, which is used in `replace_tensors`.
    """

    from aioway.t import mode_off

    def maybe_replace(item):
        if not isinstance(item, torch.Tensor):
            return NotImplemented

        return replace(item)

    with mode_off():
        return tree_map_memo(obj, maybe_replace)


def is_tensor_container(obj) -> bool:
    """
    Check if the `obj` given is considered a tensor container.
    """

    return tree_all(lambda x: isinstance(x, torch.Tensor), obj)
