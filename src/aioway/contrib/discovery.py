# Copyright (c) AIoWay Authors - All Rights Reserved

"Discover the domain of data: data -> domain."

import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway.cmds.nets import IntItemDomain, float_list, int_list
from aioway.t import (
    Domain,
    ScoreTensorDomain,
    TDictDomain,
    parse_attr,
    sized_of,
    tcol_to_tdict,
)

__all__ = ["batch_domain", "iter_domain", "tensor_domain"]


def batch_domain(item: typing.Any, /) -> Domain:
    """
    The domain of a batch, where the first dimension is the batch.

    `NotImplemented` for unknown data.
    """

    if isinstance(item, torch.Tensor):
        return tensor_domain(item)

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


def tensor_domain(tensor: torch.Tensor, /) -> Domain:
    """
    The domain of a batched tensor, `[B, ...]`, by the first rule that matches.

    Ranges are not discovered, as they would differ batch by batch.
    """

    for rule in _TENSOR_RULES:
        if (domain := rule(tensor)) is not NotImplemented:
            return domain

    return sized_of(*tensor.shape[1:])


def _float_item(tensor: torch.Tensor) -> Domain:
    if tensor.ndim == 1 and parse_attr(tensor).dtype.family == "float":
        return ScoreTensorDomain()

    return NotImplemented


def _float_list(tensor: torch.Tensor) -> Domain:
    if tensor.ndim > 1 and parse_attr(tensor).dtype.family == "float":
        return float_list(*tensor.shape[1:])

    return NotImplemented


def _int_item(tensor: torch.Tensor) -> Domain:
    if tensor.ndim == 1 and parse_attr(tensor).dtype.family == "int":
        return IntItemDomain()

    return NotImplemented


def _int_list(tensor: torch.Tensor) -> Domain:
    if tensor.ndim > 1 and parse_attr(tensor).dtype.family == "int":
        return int_list(*tensor.shape[1:])

    return NotImplemented


_TENSOR_RULES: list[cabc.Callable[[torch.Tensor], Domain]] = [
    _float_item,
    _float_list,
    _int_item,
    _int_list,
]
"The rules for `tensor_domain`, tried in order, the first match wins."
