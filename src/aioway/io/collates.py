# Copyright (c) AIoWay Authors - All Rights Reserved

import functools
import torch
from collections import abc as cabc

__all__ = ["chunk_collate", "chunk_collate_decompose_dim"]

type CollateFunc = cabc.Callable[[list[torch.Tensor]], torch.Tensor]


def chunk_collate_decompose_dim(max_len: int, dim: int) -> CollateFunc:
    """
    Instead of decomposing the 0th dimension, decompose the `dim` dimension.

    Conceptually, we collate in the `nth` dimesnion, and the decompose that:

    1. Slice the tensor [..., dim, ...] into [..., batch * max_len, ...] and drop remainder.
    2. Decompose the tensor [..., batch * max_len, ...] into [..., batch, max_len, ...].
    3. Permute the tensor [batch, ..., max_len, ...].

    Args:
        max_len: The maximum length in the frame dimension to chunk.

    Returns:
        A function that can be passed to `data.DataLoader`'s `collate_fn`.

        The function takes in a list of `torch.Tensor,
        it decomposes the first dimension of the tensor,
        and outputs the tensor of shape [batch_size, max_len, ...],
        where ... is the dimensions outside of last dimension unchanged.
    """

    def collate(samples: list[torch.Tensor]) -> torch.Tensor:
        # Make `nth` positive.
        [ndim] = {tensor.ndim for tensor in samples}
        nth = dim % ndim

        merged = torch.cat(samples, dim=nth)
        shape = merged.shape
        batch_size = shape[nth] // max_len

        # Slice then decompose nth dimension.
        merged = merged.narrow(nth, 0, batch_size * max_len)
        new_shape = *shape[:nth], batch_size, max_len, *shape[nth + 1 :]
        merged = merged.view(*new_shape)

        # Move batch dim to the first, leave rest unchanged.
        merged = merged.permute(
            nth, *range(0, nth), nth + 1, *range(nth + 2, merged.ndim)
        )
        return merged

    return collate


def chunk_collate(max_len: int):
    "Equivalent to `chunk_collate_decompose_dim(max_len, 0)`."

    return chunk_collate_decompose_dim(max_len, 0)
