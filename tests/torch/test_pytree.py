# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

import pytest
import torch

from aioway._utils import AnyDict
from aioway.torch import (
    find_nested_tensors,
    register_pytree_dcls,
    replace_tensors,
    tree_leaves_typed,
    tree_map_memo,
)


@register_pytree_dcls
@dcls.dataclass(frozen=True)
class NestedTensors:
    lists: list[torch.Tensor]
    dicts: dict[str, list[torch.Tensor]]


@register_pytree_dcls
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

    yield [1]
    yield {"a": a, "b": 1}
    nnt = NotNestedTensors([a, b, c, 1], {"g": 4})
    yield [nnt]


@pytest.fixture(params=_not_nested())
def not_nested(request):
    return request.param


def test_nested_pure(nested):
    result = list(find_nested_tensors(nested))
    assert result
    assert all(isinstance(t, torch.Tensor) for t in result)


def test_nested_impure(not_nested):
    result = list(find_nested_tensors(not_nested))
    assert all(isinstance(t, torch.Tensor) for t in result)


def test_find_order_and_identity():
    a, b, c = torch.tensor(1), torch.tensor(2), torch.tensor(3)
    result = list(find_nested_tensors([a, (b,), {"c": c}]))
    assert len(result) == 3
    assert all(x is y for x, y in zip(result, [a, b, c]))


def test_find_skips_non_tensors():
    a = torch.tensor(1)
    result = list(find_nested_tensors([1, "x", None, a, 2.0]))
    assert len(result) == 1
    assert result[0] is a


def test_find_keeps_duplicates():
    a = torch.tensor(1)
    assert len(list(find_nested_tensors([a, a]))) == 2


@pytest.mark.parametrize("obj", [1, "x", None, [], {}, ()])
def test_find_nothing(obj):
    assert list(find_nested_tensors(obj)) == []


def test_leaves_typed_filters():
    assert list(tree_leaves_typed([1, "a", [2, "b"]], int)) == [1, 2]


def test_leaves_typed_multiple_types():
    assert list(tree_leaves_typed([1, "a", 2.0], int, str)) == [1, "a"]


def test_leaves_typed_stops_at_container_type():
    # Matching a container type yields the container itself, not its contents.
    assert list(tree_leaves_typed([(1, 2), 3], tuple)) == [(1, 2)]


def _times_ten(x):
    return x * 10 if isinstance(x, int) else NotImplemented


def test_map_memo_replaces():
    assert tree_map_memo([1, "a", {"k": 2}], _times_ten) == [10, "a", {"k": 20}]


def test_map_memo_keeps_structure():
    result = tree_map_memo((1, [2]), _times_ten)
    assert result == (10, [20])
    assert isinstance(result, tuple)


def test_map_memo_calls_once_per_item():
    calls = []

    def replace(x):
        calls.append(x)
        return x

    tree_map_memo([1, 1, 2], replace)
    assert calls == [1, 2]


def test_map_memo_uses_given_memo():
    memo = AnyDict()
    tree_map_memo([1], _times_ten, memo)
    assert memo[1] == 10


def test_replace_tensors():
    a = torch.tensor([1, 2])
    result = replace_tensors({"a": a, "n": 3}, lambda t: t * 2)
    assert torch.equal(result["a"], torch.tensor([2, 4]))
    assert result["n"] == 3


def test_replace_same_tensor_once():
    a = torch.tensor(1)
    result = replace_tensors([a, a], lambda t: t.clone())
    assert result[0] is result[1]
    assert result[0] is not a
