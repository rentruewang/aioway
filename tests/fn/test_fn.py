# Copyright (c) AIoWay Authors - All Rights Reserved


import pytest
import torch

from aioway.fn import TorchDispatchStack, TorchFunctionStack, track_fn


@pytest.fixture
def a():
    return torch.randn(4)


@pytest.fixture
def b():
    return torch.randn(4)


def test_call(a: torch.Tensor, b: torch.Tensor):
    with (
        track_fn() as [func_hist, dis_hist],
        TorchFunctionStack() as funcs,
        TorchDispatchStack() as ops,
    ):
        result = a + b

    assert result.ndim == 1
    assert len(func_hist)
    assert len(dis_hist)

    # After calling so of course it's empty
    assert not len(funcs.stack)
    assert not len(ops.stack)

    assert dis_hist.memory()
    assert str(dis_hist[0])
