# Copyright (c) AIoWay Authors - All Rights Reserved


import pytest

from aioway.fate import Fate
from aioway.fn import (
    FateFn,
    Fn,
    HistoryTGraph,
    MightFn,
    NnFwdFn,
    NnInitFn,
    TDisFn,
    TensorInput,
    TFuncFn,
    TorchThunk,
)
from aioway.might import Might


def _fn_cls():
    yield FateFn
    yield TDisFn
    yield TFuncFn
    yield MightFn
    yield NnInitFn
    yield NnFwdFn

    yield Fate
    yield Might


def _input_cls():
    yield HistoryTGraph
    yield TorchThunk
    yield FateFn
    yield NnFwdFn
    yield FateFn
    yield Fate


@pytest.fixture(params=_fn_cls())
def fn_cls(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=_input_cls())
def input_cls(request: pytest.FixtureRequest):
    return request.param


def test_fn_subclass(fn_cls):
    assert isinstance(fn_cls, type)
    assert issubclass(fn_cls, Fn)


def test_inputs_subclass(input_cls):
    assert isinstance(input_cls, type)
    assert issubclass(input_cls, TensorInput)
