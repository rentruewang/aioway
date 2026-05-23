# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

import pytest
import torch

from aioway._utils import find_nested_tensors


@dcls.dataclass(frozen=True)
class NestedTensors:
    lists: list[torch.Tensor]
    dicts: dict[str, list[torch.Tensor]]


@dcls.dataclass(frozen=True)
class NotNestedTensors:
    lists: list[typing.Any]
    dicts: dict[str, typing.Any]


def _nested():
    a = torch.tensor([3])
    b = torch.tensor([4])
    c = torch.tensor([5, 6, 7])
    d = torch.tensor([8, 9])

    yield a
    yield b
    yield c
    yield d
    yield a, b, c, d
    yield {"a": a, "b": b}
    yield [a, (b,), [c, d]]

    nt = NestedTensors([a], {"b": [b], "cd": [c, d]})
    yield nt
    yield [nt, {"nt": nt}]


@pytest.fixture(params=_nested())
def nested(request):
    return request.param


def _not_nested():
    a = torch.tensor([3])
    b = torch.tensor([4])
    c = torch.tensor([5, 6, 7])
    d = torch.tensor([8, 9])

    yield [1]
    yield {"a": a, "b": 1}
    nnt = NotNestedTensors([a, b, c, 1], {"g": 4})
    yield [nnt]


@pytest.fixture(params=_not_nested())
def not_nested(request):
    return request.param


def test_nested_pure(nested):
    result = set(find_nested_tensors(nested))
    assert all(isinstance(t, torch.Tensor) for t in result)


def test_nested_impure(not_nested):
    result = set(find_nested_tensors(not_nested))
    assert all(isinstance(t, torch.Tensor) for t in result)


def test_nested_impure_fail(not_nested):
    with pytest.raises(ValueError):
        _ = set(find_nested_tensors(nested, only_tensors=True))
