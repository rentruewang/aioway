# Copyright (c) AIoWay Authors - All Rights Reserved

"Decomposing objects for inspection and debugging."

import dataclasses as dcls
import functools
from collections import abc as cabc

import numpy as np
import pandas as pd
import torch

__all__ = ["replace_tensors", "find_nested_tensors", "NestedFinder"]


def replace_tensors(
    obj: object, replace: cabc.Callable[[torch.Tensor], object]
) -> object:
    from aioway.fn import torch_function_off

    with torch_function_off():
        return _replace_tensors(obj, replace)


def _replace_tensors(
    obj: object, replace: cabc.Callable[[torch.Tensor], object]
) -> object:
    """
    Replace tensors whenever encountered with the given function.

    This function has the `__torch_function__` disabled in the scope of the rendering,
    because it can mess with attribute access, which oftentimes means that
    this function fails also during debugging if `__torch_function__` is not disabled.
    Caused by `.device` / `.shape` / `.dtype` calls, which is used in `replace_tensors`.
    """

    if isinstance(obj, torch.Tensor):
        return replace(obj)

    if isinstance(obj, np.ndarray | pd.DataFrame):
        return obj

    if isinstance(obj, cabc.Sequence):
        return [_replace_tensors(elem, replace) for elem in obj]

    if isinstance(obj, cabc.Mapping):
        return {key: _replace_tensors(elem, replace) for key, elem in obj.items()}

    if dcls.is_dataclass(obj):
        return _replace_tensors(_dataclass_as_dict(obj), replace)

    return obj


@dcls.dataclass(frozen=True)
class NestedFinder[T]:
    """
    Find the desired objects that is possibly nested.
    """

    target: type[T]
    """
    The target type to search for.
    """

    block_items: cabc.Sequence[object] = ()
    """
    Do not recurse into these objects.
    """

    block_types: cabc.Sequence[type] = ()
    """
    Do not recurse into objects of these types.
    """

    def __post_init__(self):
        if self.target in self.block_types:
            raise ValueError(
                f"You specified the target type {self.target} in the block list {self.block_types}."
            )

    def __call__(self, obj: object) -> cabc.Iterator[T]:
        if isinstance(obj, self.target):
            yield obj
            return

        if obj in self.block_items:
            return

        if isinstance(obj, self._block_types_list):
            return

        if isinstance(obj, cabc.Sequence):
            for elem in obj:
                yield from self(elem)

        if isinstance(obj, cabc.Mapping):
            for elem in obj.values():
                yield from self(elem)
            return

        # If it's a dataclass, decompose.
        if dcls.is_dataclass(obj):
            yield from self(_dataclass_as_dict(obj))
            return

    @functools.cached_property
    def _block_types_list(self) -> tuple[type, ...]:
        return tuple(self.block_types)


def find_nested_tensors(obj: object) -> cabc.Iterator[torch.Tensor]:
    """
    Find and unpack tensors from containers.
    """

    finder = NestedFinder(
        target=torch.Tensor,
        block_types=[int, float, bool, str, np.ndarray, pd.DataFrame],
    )

    yield from finder(obj)


def _dataclass_as_dict(obj: object):
    assert dcls.is_dataclass(obj), "Only handles dataclass objects."
    fields = dcls.fields(obj)
    return {field.name: getattr(obj, field.name) for field in fields}
