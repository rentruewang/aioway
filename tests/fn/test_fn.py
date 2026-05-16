# Copyright (c) AIoWay Authors - All Rights Reserved


import pytest
import torch

from aioway.fate import Fate
from aioway.fn import (
    FateFn,
    Fn,
    MightFn,
    NnForwardFn,
    NnInitFn,
    TDispatchFn,
    TFunctionFn,
    track_fn,
)
from aioway.might import Might


def _fn_cls():
    yield FateFn
    yield TDispatchFn
    yield TFunctionFn
    yield MightFn
    yield NnInitFn
    yield NnForwardFn

    yield Fate
    yield Might


@pytest.fixture(params=_fn_cls())
def fn_cls(request):
    return request.param


def test_fn_subclass(fn_cls):
    assert isinstance(fn_cls, type)
    assert issubclass(fn_cls, Fn)
