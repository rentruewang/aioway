# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

import numpy as np
import pytest
import torch

from aioway.schemas import DType


@dcls.dataclass(frozen=True)
class _CaseChecker:
    original: typing.Any
    torch_dtype: torch.dtype

    @property
    def dtype(self):
        return DType.parse(self.original)

    @property
    def rhs(self):
        return DType(self.torch_dtype)

    def check_parse(self):
        assert isinstance(self.dtype, DType)
        assert self.dtype == self.rhs

    def check_eq(self):
        assert self.dtype == self.torch_dtype
        assert self.original == self.rhs


def _golden():
    for dtype, torch_dtype in _dtypes():
        yield _CaseChecker(original=dtype, torch_dtype=torch_dtype)


def _dtypes():
    yield "float16", torch.float16
    yield np.dtype("float16"), torch.float16
    yield torch.float16, torch.float16

    yield "float32", torch.float32
    yield np.dtype("float32"), torch.float32
    yield torch.float32, torch.float32

    yield "float64", torch.float64
    yield np.dtype("float64"), torch.float64
    yield torch.float64, torch.float64

    yield "int8", torch.int8
    yield np.dtype("int8"), torch.int8
    yield torch.int8, torch.int8

    yield "int16", torch.int16
    yield np.dtype("int16"), torch.int16
    yield torch.int16, torch.int16

    yield "int32", torch.int32
    yield np.dtype("int32"), torch.int32
    yield torch.int32, torch.int32

    yield "int64", torch.int64
    yield np.dtype("int64"), torch.int64
    yield torch.int64, torch.int64

    yield "bool", torch.bool
    yield np.dtype("bool"), torch.bool
    yield torch.bool, torch.bool


@pytest.fixture(params=_golden())
def golden(request: pytest.FixtureRequest) -> _CaseChecker:
    return request.param


def test_dtype_parse(golden: _CaseChecker):
    golden.check_parse()


def test_dtype_eq(golden: _CaseChecker):
    golden.check_eq()
