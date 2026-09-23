# Copyright (c) AIoWay Authors - All Rights Reserved

"Some fake mode guards."

import functools
import typing
from collections import abc as cabc

import tensordict as td
import torch
from torch._subclasses import fake_tensor as ft

from aioway.torch._utils import tcol_to_tdict
from aioway.torch.visitors import TorchVisitor

from .contexts import fake_mode

__all__ = [
    "is_fake",
    "is_fake_tensor",
    "is_real",
    "is_real_tensor",
    "to_fake",
    "clone_fake",
]


@functools.cache
def _to_fake_converter():
    to_fake_tdict = lambda item: td.from_dict(_to_fake_dict(item))
    to_fake_seq = lambda item: [to_fake(elem) for elem in item]

    return TorchVisitor(
        tensor=_to_fake_tensor,
        tdict=to_fake_tdict,
        tcls=_to_fake_tcls,
        mapping=_to_fake_dict,
        sequence=to_fake_seq,
        default=lambda item: item,
    )


def to_fake[C](item: C) -> C:
    "Convert an item to its fake counterpart."

    return _to_fake_converter()(item)


@functools.cache
def _is_fake_converter() -> TorchVisitor[bool]:
    return TorchVisitor(
        tensor=is_fake_tensor,
        tdict=_is_fake_tcol,
        tcls=_is_fake_tcol,
        mapping=lambda item: _is_fake_iter(item.values()),
        sequence=_is_fake_iter,
        default=lambda _: False,
    )


def is_fake(item) -> bool:
    """
    Check if the item is fake.
    """

    return _is_fake_converter()(item)


def is_real(item) -> bool:
    "Check if the item is a real one."

    return is_fake(item) != True


def is_real_tensor(item) -> bool:
    return isinstance(item, torch.Tensor) and not is_fake_tensor(item)


@functools.cache
def _clone_fake_converter() -> TorchVisitor:
    clone = lambda x: x.clone()
    return TorchVisitor(
        tensor=clone,
        tdict=clone,
        tcls=clone,
        mapping=lambda item: {key: clone_fake(val) for key, val in item.items()},
        sequence=lambda item: [clone_fake(val) for val in item],
        default=lambda item: item,
    )


def clone_fake[T](item: T) -> T:
    """
    Call `.clone()` on `torch` / `tensordict` fake values.

    This is useful in changing the `id` of fake values for uniqueness analysis.
    """

    if not is_fake(item):
        return item

    return _clone_fake(item)


def _clone_fake(obj: typing.Any) -> typing.Any:
    if isinstance(obj, torch.Tensor):
        return obj.clone()

    if td.is_tensor_collection(obj):
        return obj.clone()

    if isinstance(obj, cabc.Mapping):
        return {key: clone_fake(val) for key, val in obj.items()}

    if isinstance(obj, cabc.Iterable):
        return [clone_fake(elem) for elem in obj]

    raise TypeError(f"Unknown type: {type(obj)=}.")


def _is_fake_tcol(item) -> bool:
    tdict = tcol_to_tdict(item)
    return _is_fake_iter(tdict.values())


def _is_fake_iter(item: cabc.Iterable):
    return any(is_fake(val) for val in item)


def _to_fake_tcls(item):
    tdict = tcol_to_tdict(item)
    mapping = _to_fake_dict(tdict)
    return type(item)(**mapping)


def _to_fake_tensor(tensor: torch.Tensor) -> ft.FakeTensor:
    """
    Move a possibly real tensor to a fake torch.Tensor
    """

    if is_fake_tensor(tensor):
        return tensor

    with fake_mode() as mode:
        converter = mode.fake_tensor_converter
        return converter.from_real_tensor(mode, tensor)


def is_fake_tensor(tensor: torch.Tensor) -> typing.TypeIs[ft.FakeTensor]:
    # All fake tensors are of this type.
    return isinstance(tensor, ft.FakeTensor)


def _to_fake_dict(
    tdict: cabc.Mapping[str, typing.Any],
) -> dict[str, typing.Any]:
    return {key: to_fake(val) for key, val in tdict.items()}
