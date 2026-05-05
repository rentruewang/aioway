# Copyright (c) AIoWay Authors - All Rights Reserved

from collections import abc as cabc

import tensordict as td
import torch

__all__ = ["tdict_rename", "tdict_all_equal", "replace_tensors", "find_nested_tensors"]


def tdict_rename(tdict: td.TensorDict, **renames: str):
    return td.TensorDict({renames.get(key, key): value for key, value in tdict.items()})


def tdict_all_equal(left: td.TensorDict, right: td.TensorDict, /):
    if left.keys() != right.keys():
        return False

    eq: td.TensorDict = left == right
    return eq.all()


def replace_tensors(
    obj: object, replace: cabc.Callable[[torch.Tensor], object]
) -> object:
    """
    Replace tensors whenever encountered with the given function.
    """

    if isinstance(obj, torch.Tensor):
        return replace(obj)

    if isinstance(obj, cabc.Sequence):
        return [replace_tensors(elem, replace) for elem in obj]

    if isinstance(obj, cabc.Mapping):
        return {key: replace_tensors(elem, replace) for key, elem in obj.items()}

    return obj


def find_nested_tensors(obj: object) -> cabc.Iterator[torch.Tensor]:
    """
    Find and unpack tensors from containers.
    """

    if isinstance(obj, torch.Tensor):
        yield obj
        return

    if isinstance(obj, cabc.Sequence):
        for elem in obj:
            yield from find_nested_tensors(elem)
        return

    if isinstance(obj, cabc.Mapping):
        for elem in obj.values():
            yield from find_nested_tensors(elem)
        return
