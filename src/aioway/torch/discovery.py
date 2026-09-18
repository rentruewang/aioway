# Copyright (c) AIoWay Authors - All Rights Reserved

import functools
import typing
from collections import abc as cabc

import tensordict as td
import torch
from torchrl.data import tensor_specs as tspecs

from ._utils import tcol_to_tdict
from .overrides import is_fake
from .tspecs import TSpec, default_coerce

__all__ = ["batch_tspec", "iter_tspec"]


def batch_tspec(item: object, /) -> tspecs.TensorSpec:
    if isinstance(item, torch.Tensor):
        return _tensor_tspec(item)

    if td.is_tensor_collection(item):
        tdict = tcol_to_tdict(item)
        return _tdict_tspec(tdict)

    return NotImplemented


def iter_tspec(stream: cabc.Iterable[typing.Any], /) -> TSpec:
    "Get a tspec over a stream."

    tspecs = [batch_tspec(elem) for elem in stream]
    return functools.reduce(default_coerce, tspecs)


def _tensor_tspec(tensor: torch.Tensor, /) -> tspecs.TensorSpec:
    """
    Convert `torch.Tensor` to `TSpec`.
    """

    shape = tensor.shape[1:]

    if tensor.dtype == torch.bool:
        return tspecs.Binary(shape=shape)

    if is_fake(tensor):
        return tspecs.Unbounded(shape=shape, dtype=tensor.dtype)

    return tspecs.Bounded(
        low=tensor.min(), high=tensor.max(), shape=shape, dtype=tensor.dtype
    )


def _tdict_tspec(tdict: td.TensorDict, /) -> tspecs.Composite:
    result: dict[str, tspecs.TensorSpec] = {}

    for key, val in tdict.items():
        tspec = batch_tspec(val)
        result[key] = tspec

    return tspecs.Composite(result)
