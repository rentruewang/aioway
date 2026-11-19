# Copyright (c) AIoWay Authors - All Rights Reserved

"Discover the domain of data: data -> domain."

import typing
from collections import abc as cabc

import tensordict as td
import torch

from ._utils import tcol_to_tdict
from .domains import Domain, TDictDomain, sized_of

__all__ = ["batch_domain", "iter_domain"]


def batch_domain(item: typing.Any, /) -> Domain:
    """
    The domain of a batch, where the first dimension is the batch.

    Only shapes are discovered, `NotImplemented` for unknown data.
    """

    if isinstance(item, torch.Tensor):
        return sized_of(*item.shape[1:])

    if td.is_tensor_collection(item):
        tdict = tcol_to_tdict(item)
        return TDictDomain({key: batch_domain(val) for key, val in tdict.items()})

    return NotImplemented


def iter_domain(stream: cabc.Iterable[typing.Any], /) -> Domain:
    "The domain shared by every batch in `stream`."

    domains = {batch_domain(item) for item in stream}

    if len(domains) != 1:
        raise ValueError(f"Batches should share 1 domain, got {domains=}.")

    [domain] = domains
    return domain
