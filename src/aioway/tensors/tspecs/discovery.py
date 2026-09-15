# Copyright (c) AIoWay Authors - All Rights Reserved

import tensordict as td
import torch
from torch.utils import data as dutils
from torchrl.data import tensor_specs as tspecs

from aioway.tensors.fake import is_fake

from .tspecs import TSpec

__all__ = ["tensor_tspec", "tensordict_tspec", "tensorclass_tspec", "dataloader_tspec"]


def tensor_tspec(tensor: torch.Tensor) -> TSpec:
    """
    Convert `torch.Tensor` to `TSpec`.
    """

    if tensor.dtype == torch.bool:
        return tspecs.Binary(shape=tensor.shape)

    if is_fake(tensor):
        return tspecs.Unbounded(shape=tensor.shape, dtype=tensor.dtype)
    else:
        return tspecs.Bounded(
            low=tensor.min(),
            high=tensor.max(),
            shape=tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )


def tensordict_tspec(tensor: td.TensorDict) -> TSpec:
    raise NotImplementedError


def tensorclass_tspec(tensor: type) -> TSpec:
    if not isinstance(tensor, type):
        raise TypeError(f"Only accepts types. Got {tensor=}.")

    raise NotImplementedError


def dataloader_tspec(dset: dutils.DataLoader) -> TSpec:
    raise NotImplementedError
