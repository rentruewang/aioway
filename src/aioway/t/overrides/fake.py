# Copyright (c) AIoWay Authors - All Rights Reserved

"Some fake mode guards."

import typing

import torch
from torch._subclasses import fake_tensor as ft
from torch.utils import _pytree as pyt

from .contexts import fake_mode

__all__ = [
    "has_fake",
    "all_fake",
    "has_real",
    "all_real",
    "is_fake_tensor",
    "is_real_tensor",
    "to_fake",
    "clone_fake",
]


def to_fake[C](item: C) -> C:
    "Convert an item to its fake counterpart."

    return pyt.tree_map_only(torch.Tensor, func=_to_fake_tensor, tree=item)


def has_fake(item) -> bool:
    """
    Check if the item is fake.
    """

    for elem in pyt.tree_leaves(item):
        # This already checks if it's a tensor.
        if is_fake_tensor(elem):
            return True

    return False


def all_fake(item) -> bool:
    """
    Check if all the tesnors in here are all fake tensors.
    """

    for elem in pyt.tree_leaves(item):
        # May have other values not tensor, only care about tensors.
        if is_real_tensor(elem):
            return False

    return True


def has_real(item) -> bool:
    "Check if the item is a real one."

    return not all_fake(item)


def all_real(item) -> bool:
    "Check if the item is a real one."

    return not has_fake(item)


def is_real_tensor(item) -> bool:
    return isinstance(item, torch.Tensor) and not is_fake_tensor(item)


def clone_fake[T](item: T) -> T:
    """
    Call `.clone()` on `torch` / `tensordict` fake values.

    This is useful in changing the `id` of fake values for uniqueness analysis.
    """

    return pyt.tree_map_only(torch.Tensor, func=_clone_fake_tensor, tree=item)


def _to_fake_tensor(tensor: torch.Tensor) -> ft.FakeTensor:
    """
    Move a possibly real tensor to a fake torch.Tensor
    """

    if is_fake_tensor(tensor):
        return tensor

    with fake_mode() as mode:
        converter = mode.fake_tensor_converter
        return converter.from_real_tensor(mode, tensor)


def _clone_fake_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if is_fake_tensor(tensor):
        return tensor.clone()
    else:
        return tensor


def is_fake_tensor(tensor: torch.Tensor) -> typing.TypeIs[ft.FakeTensor]:
    # All fake tensors are of this type.
    return isinstance(tensor, ft.FakeTensor)
