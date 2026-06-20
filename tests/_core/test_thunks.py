# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway._core import TensorInput, TensorThunk, Thunk
from aioway.modes import (
    Aten,
    AtenThunk,
    HistTensorGraph,
    NnFwdThunk,
    NnInitThunk,
    TorchDispThunk,
    TorchFuncThunk,
)


def _fn_cls():
    yield AtenThunk
    yield TorchDispThunk
    yield TorchFuncThunk
    yield NnInitThunk
    yield NnFwdThunk

    yield Aten


def _input_cls():
    yield HistTensorGraph
    yield TensorThunk
    yield AtenThunk
    yield NnFwdThunk
    yield AtenThunk
    yield Aten


@pytest.fixture(params=_fn_cls())
def fn_cls(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=_input_cls())
def input_cls(request: pytest.FixtureRequest):
    return request.param


def test_fn_subclass(fn_cls):
    assert isinstance(fn_cls, type)
    assert issubclass(fn_cls, Thunk)


def test_inputs_subclass(input_cls):
    assert isinstance(input_cls, type)
    assert issubclass(input_cls, TensorInput)
