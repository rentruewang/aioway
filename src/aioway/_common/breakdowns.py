# Copyright (c) AIoWay Authors - All Rights Reserved

"Decomposing objects for inspection and debugging."

import dataclasses as dcls
from collections import abc as cabc

import numpy as np
import torch

__all__ = ["replace_tensors", "find_nested_tensors"]


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

    if _is_primitive(obj):
        return obj

    if isinstance(obj, cabc.Sequence):
        return [_replace_tensors(elem, replace) for elem in obj]

    if isinstance(obj, cabc.Mapping):
        return {key: _replace_tensors(elem, replace) for key, elem in obj.items()}

    return obj


def find_nested_tensors(obj: object) -> cabc.Iterator[torch.Tensor]:
    """
    Find and unpack tensors from containers.
    """

    if isinstance(obj, torch.Tensor):
        yield obj
        return

    if _is_primitive(obj):
        return

    if isinstance(obj, cabc.Sequence):
        for elem in obj:
            yield from find_nested_tensors(elem)
        return

    if isinstance(obj, cabc.Mapping):
        for elem in obj.values():
            yield from find_nested_tensors(elem)
        return

    # If it's a dataclass, decompose.
    if dcls.is_dataclass(obj):
        fields = dcls.fields(obj)
        yield from find_nested_tensors(
            {field.name: getattr(obj, field.name) for field in fields}
        )


def _is_primitive(obj: object) -> bool:
    if obj in [None, NotImplemented, ...]:
        return True

    if isinstance(obj, int | float | bool | str | np.ndarray):
        return True

    return False
