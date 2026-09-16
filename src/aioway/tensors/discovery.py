# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway.tensors import TSpec
import tensordict as td, typing
import torch
from torchrl.data import tensor_specs as tspecs
from collections import abc as cabc
from ._utils import tcol_to_tdict
from .fake import is_fake
import functools
from .tspecs import default_coerce

__all__ = ["batch_tspec"]


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

    if tensor.dtype == torch.bool:
        return tspecs.Binary(shape=tensor.shape)

    if is_fake(tensor):
        return tspecs.Unbounded(shape=tensor.shape, dtype=tensor.dtype)

    return tspecs.Bounded(
        low=tensor.min(), high=tensor.max(), shape=tensor.shape, dtype=tensor.dtype
    )


def _tdict_tspec(tdict: td.TensorDict, /) -> tspecs.Composite:
    result: dict[str, tspecs.TensorSpec] = {}

    for key, val in tdict.items():
        tspec = batch_tspec(val)
        result[key] = tspec

    return tspecs.Composite(result)
