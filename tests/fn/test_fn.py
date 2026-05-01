# Copyright (c) AIoWay Authors - All Rights Reserved


import pytest
import torch

from aioway.fn import TorchDispatchStack, TorchFunctionStack, track_dispatch_fn


@pytest.fixture
def a():
    return torch.randn(4)


@pytest.fixture
def b():
    return torch.randn(4)


def test_call(a: torch.Tensor, b: torch.Tensor):
    with (
        track_dispatch_fn() as calls,
        TorchFunctionStack() as funcs,
        TorchDispatchStack() as ops,
    ):
        result = a + b

    assert result.ndim == 1
    assert len(calls)

    # After calling so of course it's empty
    assert not len(funcs.stack)
    assert not len(ops.stack)

    assert calls.memory()
    assert str(calls[0])
