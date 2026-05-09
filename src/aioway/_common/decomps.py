# Copyright (c) AIoWay Authors - All Rights Reserved

"Decomposing objects for inspection and debugging."

import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

import numpy as np
import pandas as pd
import tensordict as td
import torch

__all__ = [
    "replace_tensors",
    "find_nested_tensors",
    "Decomposer",
    "DecomposeCheck",
    "DecompStep",
    "DecompSeq",
    "DecompMap",
    "DecompDcls",
    "DECOMP_BLOCK_ITEMS",
    "DECOMP_BLOCK_TYPES",
]

DECOMP_BLOCK_ITEMS = None, NotImplemented, ..., True, False
DECOMP_BLOCK_TYPES = (
    int,
    float,
    bool,
    str,
    np.ndarray,
    pd.DataFrame,
    torch.Tensor,
    td.TensorDict,
)


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

    if isinstance(obj, DECOMP_BLOCK_TYPES):
        return obj

    if isinstance(obj, cabc.Sequence):
        return [_replace_tensors(elem, replace) for elem in obj]

    if isinstance(obj, cabc.Mapping):
        return {key: _replace_tensors(elem, replace) for key, elem in obj.items()}

    if dcls.is_dataclass(obj):
        return _replace_tensors(_dataclass_as_dict(obj), replace)

    return obj


@typing.runtime_checkable
class DecomposeCheck(typing.Protocol):
    def __call__(self, obj: object, /) -> bool: ...


@typing.runtime_checkable
class DecompStep(typing.Protocol):
    def handles(self, obj, /) -> bool: ...
    def decompose(self, obj, /) -> cabc.Iterable[typing.Any]: ...
@functools.cache
def _ids_of(seq: cabc.Sequence[typing.Any]) -> list[int]:
    return [id(item) for item in seq]


def default_stop_decompose(obj: object) -> bool:
    return id(obj) in _ids_of(DECOMP_BLOCK_ITEMS) or isinstance(obj, DECOMP_BLOCK_TYPES)


class DecompSeq:
    def handles(self, obj: object, /):
        return isinstance(obj, cabc.Sequence)

    def decompose(self, obj, /) -> cabc.Iterable[typing.Any]:
        yield from obj


class DecompMap:
    def handles(self, obj, /):
        return isinstance(obj, cabc.Mapping)

    def decompose(self, obj, /) -> cabc.Iterable[typing.Any]:
        yield from obj.values()


class DecompDcls:
    def handles(self, obj: object, /):
        return dcls.is_dataclass(obj)

    def decompose(self, obj, /) -> cabc.Iterable[typing.Any]:
        yield from _dataclass_as_dict(obj).values()


@dcls.dataclass(frozen=True)
class Decomposer:
    """
    Find the desired objects that is possibly nested.
    """

    target: DecomposeCheck
    """
    The target type to search for.
    """

    stop: DecomposeCheck = default_stop_decompose
    """
    Stop decomposing if `stop` returns `True`.
    """

    steps: cabc.Sequence[DecompStep] = DecompSeq(), DecompMap(), DecompDcls()
    """
    Steps to decompose the container object encountered.
    """

    def __post_init__(self):
        if not isinstance(self.target, DecomposeCheck):
            raise TypeError(f"{self.target=} is not callable.")

        if not isinstance(self.stop, DecomposeCheck):
            raise TypeError(f"{self.stop=} is not callable.")

        if not all(isinstance(step, DecompStep) for step in self.steps):
            raise TypeError(f"{self.steps=} is not callable.")

    def __call__(self, obj: object) -> cabc.Generator[typing.Any]:
        # This is what we are looking for.
        if self.target(obj):
            yield obj
            return

        # Do not proceed if `stop` signal is `True`.
        if self.stop(obj):
            return

        for step in self.steps:
            # Each step checks if decomposition is Ok.
            if not step.handles(obj):
                continue

            # Decompose the item, then recurse.
            for item in step.decompose(obj):
                yield from self(item)

            # Assume each decomposition is mutually exclusive.
            return


def find_nested_tensors(obj: object) -> cabc.Iterator[torch.Tensor]:
    """
    Find and unpack tensors from containers.
    """

    finder = Decomposer(target=lambda t: isinstance(t, torch.Tensor))
    yield from finder(obj)


def _dataclass_as_dict(obj: object):
    assert dcls.is_dataclass(obj), "Only handles dataclass objects."
    fields = dcls.fields(obj)
    return {field.name: getattr(obj, field.name) for field in fields}
