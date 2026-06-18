# Copyright (c) AIoWay Authors - All Rights Reserved

"Decomposing objects for inspection and debugging."

import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

import numpy as np
import pandas as pd
import torch

from .types import AnyDict

__all__ = [
    "replace_tensors",
    "decomp_flatten",
    "decomp_replace",
    "find_nested_tensors",
    "dcls_asdict",
    "decomp_block_items",
    "decomp_block_types",
    "decomp_dcls_members",
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

    from aioway.modes import mode_off

    def maybe_replace(item):
        if not isinstance(item, torch.Tensor):
            return NotImplemented

        return replace(item)

    with mode_off():
        return decomp_replace(obj, maybe_replace)


def stop_decompose(obj: object) -> bool:
    # Check if it's those primitives.
    for item in _decomp_block_items:
        if obj is item:
            return True

    return isinstance(obj, tuple(_decomp_block_types))


def decomp_replace(
    obj,
    replace: cabc.Callable[..., object],
    memo: AnyDict[typing.Any, typing.Any] | None = None,
) -> object:
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
    return _decomp_replace(obj, replace, memo)


def _decomp_replace(
    obj,
    replace: cabc.Callable[..., object],
    memo: AnyDict[typing.Any, typing.Any],
) -> typing.Any:

    # If it returns a proper value, it will be replaced (if not in `memo`).
    # Things in `memo` are prioritized.
    if (replaced := replace(obj)) is not NotImplemented:
        if obj not in memo:
            memo[obj] = replaced

        return memo[obj]

    if stop_decompose(obj):
        return obj

    if isinstance(obj, cabc.Sequence):
        return [_decomp_replace(elem, replace, memo) for elem in obj]

    if isinstance(obj, cabc.Mapping):
        return {key: _decomp_replace(elem, replace, memo) for key, elem in obj.items()}

    if not isinstance(obj, type) and dcls.is_dataclass(obj):
        obj_type: typing.Any = type(obj)
        fields = dcls_asdict(obj)
        fields = _decomp_replace(fields, replace, memo)
        return obj_type(**fields)

    return obj


def decomp_dcls_members(
    obj, types: type | tuple[type, ...]
) -> cabc.Iterator[typing.Any]:
    """
    Decompose dataclass members, 1 layer deep.
    Using this still handles e.g. type `A` having `list[A]` as members.
    """

    if not dcls.is_dataclass(obj):
        raise TypeError(f"The input {obj} should be a dataclass object.")

    yield from decomp_flatten(dcls_asdict(obj), types)


def decomp_flatten(
    obj, types: type | tuple[type, ...], /, strict: bool = False
) -> cabc.Iterator[typing.Any]:
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
