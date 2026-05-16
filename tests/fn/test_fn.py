# Copyright (c) AIoWay Authors - All Rights Reserved


import pytest

from aioway.fate import Fate
from aioway.fn import (
    FateFn,
    Fn,
    FnHistory,
    MightFn,
    NnForwardFn,
    NnInitFn,
    TDispatchFn,
    TensorInput,
    TFunctionFn,
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


def _input_cls():
    yield FnHistory


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
