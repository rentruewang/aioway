# Copyright (c) AIoWay Authors - All Rights Reserved

"The implementation for dtypes, supports different backends."

import functools
import logging
import re
import typing

import numpy as np
import torch

from aioway._common import get_tracker

__all__ = ["DType", "DTypeLike"]

LOGGER = logging.getLogger(__name__)
TRACKER = get_tracker(lambda: DType)

type DTypeFamily = typing.Literal["int", "float", "bool"]
"""
The DType strings family type.
"""


type _PrimitiveType = type[int] | type[float] | type[bool]
type DTypeLike = str | DType | _PrimitiveType | torch.dtype | np.dtype
"Types that can be converted to `Dtype` with the public `dtype` function (or `DType.parse`)."


class DType:
    r"""
    `DType` is a class supporting converting to and from
    its string representation in `aioway`, effectively supporting
    comparison and conversion between different frameworks.
    """

    def __init__(self, family: DTypeFamily, bits: int):
        if bits % 8 != 0:
            raise ValueError(f"Bits should be multiple of 8. Got bits={self._bits}.")

        self._family: DTypeFamily = family
        self._bits: int = bits

    def __str__(self):
        """
        Get the representation of the type, must be the most specialized.

        For example, the output should not be "float" due to ambiguity,
        but rather "float32" or "float64" etc.
        """

        match f := self._family:
            case "bool":
                return "bool"
            case "int" | "float":
                return f"{f}{self._bits}"
            case _:
                raise NotImplementedError(self._family)

    @typing.override
    def __getstate__(self) -> object:
        return self.family, self.bits

    def __hash__(self) -> int:
        return hash(str(self))

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: typing.Any) -> bool:
        try:
            parsed = self.parse(other)
        except ValueError:
            return NotImplemented

        # Parsing sucessful.
        return self.family == parsed.family and self.bits == parsed.bits

    @property
    def family(self) -> DTypeFamily:
        """
        The family of the dtype. For example, "float64"'s family is "float".
        """

        return self._family

    @property
    def bits(self) -> int:
        """
        The width of the dtype in bytes. Greater or equal to 1.
        """
        return self._bits

    def numpy(self) -> np.dtype:
        "Convert this to a numpy dtype."
        return np.dtype(str(self))

    def torch(self) -> torch.dtype:
        "Convert this to a torch dtype."
        return getattr(torch, str(self))

    def broadcast(self, other: DTypeLike) -> typing.Self:
        try:
            rhs = DType.parse(other)
        except ValueError:
            return NotImplemented

        np_lhs = self.numpy()
        np_rhs = rhs.numpy()

        promoted = np.result_type(np_lhs, np_rhs)
        return self.parse(promoted)

    @classmethod
    def boolean(cls) -> typing.Self:
        return cls(family="bool", bits=8)

    @classmethod
    def parse(cls, dtype: DTypeLike) -> typing.Self:
        """
        The convenient wrapper to create a `DType` from compatible types.

        Raises:
            ValueError: If we don't know how to handle the dtype.
        """

        if isinstance(dtype, cls):
            return dtype

        # Handling the basic types.
        if isinstance(dtype, type):
            return cls._parse_primitive_type(dtype)

        # Convert with regex.
        if isinstance(dtype, str):
            return cls._parse_regex(dtype)

        if isinstance(dtype, torch.dtype):
            return cls._parse_torch(dtype)

        if isinstance(dtype, np.dtype):
            return cls._parse_numpy(dtype)

        raise ValueError(f"Not sure how to handle {dtype=}.")

    @classmethod
    def _parse_primitive_type(cls, dtype: type) -> typing.Self:
        if dtype == int:
            return cls("int", 64)

        if dtype == float:
            return cls("float", 32)

        if dtype == bool:
            return cls("bool", 8)

        raise ValueError(dtype)

    @classmethod
    def _parse_regex(cls, dtype: str, /) -> typing.Self:
        """
        Create the `DType` instance from the `info` object.

        Raises:
            ValueError: If the dtyep cannot be parsed.
        """

        if m := _int_dtype().match(dtype):
            _, bits = m.groups()
            return cls(family="int", bits=int(bits) if bits else 64)

        if m := _float_dtype().match(dtype):
            _, bits = m.groups()
            return cls(family="float", bits=int(bits) if bits else 32)

        if m := _bool_dtype().match(dtype):
            return cls(family="bool", bits=8)

        raise ValueError(dtype)

    @classmethod
    @typing.no_type_check
    def _parse_torch(cls, dtype: torch.dtype, /) -> typing.Self:
        "Create a `Dtype` from a `torch.dtype`."
        if dtype == torch.bool:
            return cls.boolean()

        if dtype.is_complex:
            raise ValueError("We do not handle complex types yet!")

        # Itemsize correspond to bytes.
        bits = dtype.itemsize * 8

        # Both complex and bool are handled above.
        family: DTypeFamily = "float" if dtype.is_floating_point else "int"

        return cls(family=family, bits=bits)

    @classmethod
    def _parse_numpy(cls, dtype: np.dtype, /) -> typing.Self:
        family = _parse_numpy_family(dtype)
        bits = _parse_numpy_bits(dtype, family)

        return cls(family=family, bits=bits)


def _parse_numpy_family(dtype: np.dtype, /) -> DTypeFamily:
    "Create a `Dtype` from a `numpy.dtype`."
    if np.isdtype(dtype, "integral"):
        return "int"

    if np.isdtype(dtype, "real floating"):
        return "float"

    if np.isdtype(dtype, "bool"):
        return "bool"

    raise ValueError(f"Cannot handle numpy {dtype=}.")


def _parse_numpy_bits(dtype: np.dtype, family: DTypeFamily, /):
    match family:
        case "bool":
            return 8
        case "int":
            return np.iinfo(dtype).bits
        case "float":
            return np.finfo(dtype).bits

    raise ValueError(f"Unhandled {family=}")


@functools.cache
def _float_dtype():
    return re.compile(r"^(float)(16|32|64|128)?$", re.IGNORECASE)


@functools.cache
def _int_dtype():
    return re.compile(r"^(int)(8|16|32|64|128)?$", re.IGNORECASE)


@functools.cache
def _bool_dtype():
    return re.compile(r"^(bool)$", re.IGNORECASE)
