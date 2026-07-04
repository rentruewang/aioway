# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

import numpy as np
import pytest
import torch

from aioway.attrs import DType, DTypeFamily


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
        assert isinstance(self.dtype, DType), self.__test_case
        assert self.dtype == self.rhs, self.__test_case

    def check_eq(self):
        assert self.dtype == self.torch_dtype, self.__test_case
        assert self.original == self.rhs, self.__test_case

    def check_family(self):
        assert self.rhs.family == self.family, self.__test_case

    @property
    def family(self) -> DTypeFamily:
        if self.torch_dtype == torch.bool:
            return "bool"

        if self.torch_dtype in [torch.complex32, torch.complex64, torch.complex128]:
            return "complex"

        if self.torch_dtype in [torch.uint8, torch.uint16, torch.uint32, torch.uint64]:
            return "uint"

        if self.torch_dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            return "int"

        if self.torch_dtype in [torch.float16, torch.float32, torch.float64]:
            return "float"

        raise ValueError(f"Unknown type: {self.torch_dtype}")

    @property
    def __test_case(self):
        return f"{self.original=}, {self.rhs=}, {self.family=}"


def _golden():
    for dtype, torch_dtype in _dtypes():
        yield _CaseChecker(original=dtype, torch_dtype=torch_dtype)


def _dtypes():
    "LHS: the data to try parse. RHS: The backing field for `DType`."

    yield "float16", torch.float16
    yield np.dtype("float16"), torch.float16
    yield torch.float16, torch.float16

    yield "float", torch.float32
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

    yield "int", torch.int32
    yield "int32", torch.int32
    yield np.dtype("int32"), torch.int32
    yield torch.int32, torch.int32

    yield "long", torch.int64
    yield "int64", torch.int64
    yield np.dtype("int64"), torch.int64
    yield torch.int64, torch.int64

    yield "bool", torch.bool
    yield np.dtype("bool"), torch.bool
    yield torch.bool, torch.bool

    yield "uint8", torch.uint8
    yield np.dtype("uint8"), torch.uint8
    yield torch.uint8, torch.uint8

    yield "uint16", torch.uint16
    yield np.dtype("uint16"), torch.uint16
    yield torch.uint16, torch.uint16

    yield "uint32", torch.uint32
    yield np.dtype("uint32"), torch.uint32
    yield torch.uint32, torch.uint32

    yield "uint64", torch.uint64
    yield np.dtype("uint"), torch.uint64
    yield np.dtype("uint64"), torch.uint64
    yield torch.uint64, torch.uint64

    yield "complex32", torch.complex32
    yield torch.complex32, torch.complex32

    yield "complex64", torch.complex64
    yield np.dtype("complex64"), torch.complex64
    yield torch.complex64, torch.complex64

    yield "complex128", torch.complex128
    yield np.dtype("complex128"), torch.complex128
    yield torch.complex128, torch.complex128


@pytest.fixture(params=_golden())
def golden(request: pytest.FixtureRequest) -> _CaseChecker:
    return request.param


def test_dtype_parse(golden: _CaseChecker):
    "Check the `DType.parse` different objects."
    golden.check_parse()


def test_dtype_eq(golden: _CaseChecker):
    "Check the `==` operator of `DType`."
    golden.check_eq()


def test_dtype_family(golden: _CaseChecker):
    "Check the family of the case."
    golden.check_family()
