# Copyright (c) AIoWay Authors - All Rights Reserved

"Decomposing tensor related data structures."

import typing
from collections import abc as cabc

import torch

from aioway._utils import decomp_flatten, decomp_replace

__all__ = ["replace_tensors", "replace_tensors_with_attr", "find_nested_tensors"]


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

    from aioway.tensors.modes import mode_off

    def maybe_replace(item):
        if not isinstance(item, torch.Tensor):
            return NotImplemented

        return replace(item)

    with mode_off():
        return decomp_replace(obj, maybe_replace)


@typing.no_type_check
def replace_tensors_with_attr[T](obj: T) -> T:
    from aioway.tensors.attrs import Attr

    return replace_tensors(obj, Attr.parse)


def find_nested_tensors(
    obj: object, *, only_tensors: bool = False
) -> cabc.Iterator[torch.Tensor]:
    """
    Find and unpack tensors from containers.

    If `only_tensors` is `True`, raies an error
    if `obj` cannot be decomposed into purely tensors.
    """

    yield from decomp_flatten(obj, torch.Tensor, strict=only_tensors)
