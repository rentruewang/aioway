# Copyright (c) AIoWay Authors - All Rights Reserved

"Decomposing objects for inspection and debugging."

import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

import numpy as np
import pandas as pd
import torch

__all__ = [
    "replace_tensors",
    "decomp_flatten",
    "find_nested_tensors",
    "dcls_asdict",
    "decomp_block_items",
    "decomp_block_types",
]

_decomp_block_items: tuple[typing.Any, ...] = None, NotImplemented, ..., True, False
"The default instances to block. You could modify this."

_decomp_block_types: tuple[type, ...] = int, float, bool, str, np.ndarray, pd.DataFrame
"The default types to block. You could modify this."


@ctxl.contextmanager
def decomp_block_items(*items: typing.Any):
    global _decomp_block_items
    prev = _decomp_block_items
    _decomp_block_items = items
    try:
        yield
    finally:
        _decomp_block_items = prev


@ctxl.contextmanager
def decomp_block_types(*types: type):
    global _decomp_block_types
    prev = _decomp_block_types
    _decomp_block_types = types
    try:
        yield
    finally:
        _decomp_block_types = prev


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

    from aioway._modes import mode_off

    with mode_off():
        return decomp_replace(obj, torch.Tensor, replace)


def stop_decompose(obj: object) -> bool:
    # Check if it's those primitives.
    for item in _decomp_block_items:
        if obj is item:
            return True

    return isinstance(obj, tuple(_decomp_block_types))


def decomp_replace(
    obj,
    types: type | tuple[type, ...],
    replace: cabc.Callable[..., object],
) -> object:
    """
    Decompose and replace
    """

    if isinstance(obj, types):
        return replace(obj)

    if stop_decompose(obj):
        return obj

    if isinstance(obj, cabc.Sequence):
        return [decomp_replace(elem, types, replace) for elem in obj]

    if isinstance(obj, cabc.Mapping):
        return {key: decomp_replace(elem, types, replace) for key, elem in obj.items()}

    if dcls.is_dataclass(obj):
        obj = dcls_asdict(obj)
        return decomp_replace(obj, types, replace)

    return obj


def decomp_flatten(obj, types: type | tuple[type, ...], /, strict: bool = False):
    "Decompose the object based on the desired type."

    if isinstance(obj, types):
        yield obj
        return

    if stop_decompose(obj):
        return

    if isinstance(obj, cabc.Sequence):
        for item in obj:
            yield from decomp_flatten(item, types, strict=strict)
        return

    if isinstance(obj, cabc.Mapping):
        for val in obj.values():
            yield from decomp_flatten(val, types, strict=strict)
        return

    if dcls.is_dataclass(obj):
        obj = dcls_asdict(obj)
        yield from decomp_flatten(obj, types, strict=strict)
        return

    # Only unhandled input would reach here. If `.strict`, raise `ValueError`.
    if strict:
        raise ValueError(f"The object {obj=} is not handled.")


def find_nested_tensors(
    obj: object, *, only_tensors: bool = False
) -> cabc.Iterator[torch.Tensor]:
    """
    Find and unpack tensors from containers.

    If `only_tensors` is `True`, raies an error
    if `obj` cannot be decomposed into purely tensors.
    """

    yield from decomp_flatten(obj, torch.Tensor, strict=only_tensors)


def dcls_asdict(obj: object) -> dict[str, typing.Any]:
    "Official `asdict` fail with some custom `__getstate__`s."

    assert dcls.is_dataclass(obj), "Only handles dataclass objects."
    fields = dcls.fields(obj)
    return {field.name: getattr(obj, field.name) for field in fields}
