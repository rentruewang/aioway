# Copyright (c) AIoWay Authors - All Rights Reserved

"Extension of `pytree` from `torch`."

import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from torch.utils import _pytree as pytree

from .types import AnyDict

__all__ = ["tree_leaves_typed", "tree_map_memo", "dcls_asdict", "find_nested_tensors"]


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
            result = replace(item)

            # If it is `NotImplemented`, no replacement is made.
            if result is NotImplemented:
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


def dcls_asdict(obj: object) -> dict[str, typing.Any]:
    "Official `asdict` fail with some custom `__getstate__`s."

    assert dcls.is_dataclass(obj), "Only handles dataclass objects."
    fields = dcls.fields(obj)
    return {field.name: getattr(obj, field.name) for field in fields}


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
