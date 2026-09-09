# Copyright (c) AIoWay Authors - All Rights Reserved

"A bunch of context managers controlling the fake mode."

import dataclasses as dcls
import logging
import typing
from collections import abc as cabc

import tensordict as td
import torch
from torch._subclasses import fake_tensor as ft

from aioway._utils import dcls_asdict

from .modes import fake_mode

__all__ = ["is_fake", "is_real", "to_fake"]

LOGGER = logging.getLogger(__name__)


@typing.overload
def to_fake(item: torch.Tensor) -> ft.FakeTensor: ...


@typing.overload
def to_fake[C](item: C) -> C: ...


def to_fake(item):
    if isinstance(item, torch.Tensor):
        return _to_fake_tensor(item)

    if isinstance(item, td.TensorDict | cabc.Mapping):
        return td.from_dict(_to_fake_dict(item))

    if not isinstance(item, type) and td.is_tensor_collection(item):
        tdict = _tcol_to_tdict(item)
        mapping = _to_fake_dict(tdict)
        return type(item)(**mapping)

    if isinstance(item, cabc.Sequence):
        return [to_fake(elem) for elem in item]

    return item


@typing.overload
def is_fake(item: torch.Tensor) -> typing.TypeIs[ft.FakeTensor]: ...


@typing.overload
def is_fake(item) -> bool: ...


def is_fake(item) -> bool:
    """
    Check if the item is fake, return `NotImplemented` for non tensor object.
    """

    if isinstance(item, torch.Tensor):
        return _is_fake_tensor(item)

    # Check if it's a `td.TensorClass` or `td.TensorDict` item.
    if not isinstance(item, type) and td.is_tensor_collection(item):
        tdict = _tcol_to_tdict(item)
        return is_fake(tdict)

    if isinstance(item, cabc.Mapping):
        return is_fake(item.values())

    if isinstance(item, cabc.Iterable):
        return any(is_fake(val) for val in item)

    return False


def is_real(item) -> bool:
    "Check if the item is a real one."

    return is_fake(item) != True


def _tcol_to_tdict(item) -> td.TensorDict:
    "Convert from tensor collection to `TensorDict`."

    if not td.is_tensor_collection(item):
        raise ValueError("Not tensor collection.")

    if isinstance(item, td.TensorDict):
        return item

    assert dcls.is_dataclass(item)
    attrs = dcls_asdict(item)
    result = td.from_dict(attrs)
    assert isinstance(result, td.TensorDict)
    return result


def _to_fake_tensor(tensor: torch.Tensor) -> ft.FakeTensor:
    """
    Move a possibly real tensor to a fake torch.Tensor
    """

    if _is_fake_tensor(tensor):
        return tensor

    with fake_mode() as mode:
        converter = mode.fake_tensor_converter
        return converter.from_real_tensor(mode, tensor)


def _is_fake_tensor(tensor: torch.Tensor) -> typing.TypeIs[ft.FakeTensor]:
    # All fake tensors are of this type.
    return isinstance(tensor, ft.FakeTensor)


def _to_fake_dict(
    tdict: cabc.Mapping[str, typing.Any],
) -> dict[str, typing.Any]:
    return {key: to_fake(val) for key, val in tdict.items()}
