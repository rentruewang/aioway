# Copyright (c) AIoWay Authors - All Rights Reserved


import pytest
import torch

from aioway.fn import (
    track_dispatch_fn_mode,
    track_function_fn_mode,
)


@pytest.fixture
def a():
    return torch.randn(4)


@pytest.fixture
def b():
    return torch.randn(4)


def test_call(a: torch.Tensor, b: torch.Tensor):
    with track_dispatch_fn_mode() as calls, track_function_fn_mode() as funcs:
        result = a + b

    assert result.ndim == 1
    assert len(calls)
    assert len(funcs)
    assert calls.memory()
    assert str(calls[0])
