# Copyright (c) AIoWay Authors - All Rights Reserved

import tensordict as td
import torch
from torchrl.data import tensor_specs as tspecs

from ._utils import tcol_to_tdict
from .fake import is_fake

__all__ = ["discover_tspec"]


def discover_tspec(item: object, /) -> tspecs.TensorSpec:
    if isinstance(item, torch.Tensor):
        return _tensor_tspec(item)

    if td.is_tensor_collection(item):
        tdict = tcol_to_tdict(item)
        return _tdict_tspec(tdict)

    return NotImplemented


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
        tspec = discover_tspec(val)
        result[key] = tspec

    return tspecs.Composite(result)
