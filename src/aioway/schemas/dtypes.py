# Copyright (c) AIoWay Authors - All Rights Reserved

"The implementation for dtypes, supports different backends."

import logging
import typing

import numpy as np
import torch

from ._bases import TorchAttrBase

__all__ = ["DType", "DTypeLike", "DTypeFamily"]

LOGGER = logging.getLogger(__name__)

type DTypeFamily = typing.Literal["int", "float", "bool", "uint", "complex"]

type DTypeLike = str | DType | torch.dtype | np.dtype
"Types that can be converted to `Dtype` with the public `dtype` function (or `DType.parse`)."


class DType(TorchAttrBase[torch.dtype]):
    r"""
    `DType` is a class supporting converting to and from
    its string representation in `aioway`, effectively supporting
    comparison and conversion between different frameworks.
    """

    TYPE = torch.dtype

    @typing.override
    def __str__(self) -> str:
        """
        Get the representation of the type, must be the most specialized.
        """

        return str(self._data).removeprefix("torch.")

    @typing.override
    def __getstate__(self) -> object:
        return str(self)

    def __hash__(self) -> int:
        return hash(str(self))

    @property
    def itemsize(self) -> int:
        """
        The width of the dtype in bytes. Greater or equal to 1.
        """

        return self._data.itemsize

    @property
    def is_complex(self) -> bool:
        return self._data.is_complex

    @property
    def is_floating_point(self) -> bool:
        return self._data.is_floating_point

    @property
    def is_signed(self) -> bool:
        return self._data.is_signed

    @property
    def family(self) -> DTypeFamily:
        if self._data == torch.bool:
            return "bool"

        elif self.is_complex:
            return "complex"

        elif self.is_floating_point:
            return "float"

        elif self.is_signed:
            return "int"

        else:
            return "uint"

    def numpy(self) -> np.dtype:
        "Convert this to a numpy dtype."
        return np.dtype(str(self))

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
        return cls(torch.bool)

    @classmethod
    def parse(cls, dtype: DTypeLike) -> typing.Self:
        """
        The convenient wrapper to create a `DType` from compatible types.

        Returns:
            An instance, or `NotImplemented` if we don't know how to parse the type.
        """

        if isinstance(dtype, cls):
            return dtype

        # Convert with regex.
        if isinstance(dtype, str):
            return cls._parse_str(dtype)

        if isinstance(dtype, torch.dtype):
            return cls._parse_torch(dtype)

        if isinstance(dtype, np.dtype):
            return cls._parse_numpy(dtype)

        raise ValueError(f"Parsing failed for {dtype=}.")

    @classmethod
    def _parse_str(cls, dtype: str, /) -> typing.Self:
        """
        Create the `DType` instance from the `info` object.

        Raises:
            ValueError: If the dtyep cannot be parsed.
        """

        try:
            torch_dtype = getattr(torch, dtype)
            return cls(torch_dtype)
        except AttributeError as err:
            raise ValueError from err

    @classmethod
    @typing.no_type_check
    def _parse_torch(cls, dtype: torch.dtype, /) -> typing.Self:
        "Create a `Dtype` from a `torch.dtype`."

        return cls(dtype)

    @classmethod
    def _parse_numpy(cls, dtype: np.dtype, /) -> typing.Self:
        # Since `np.dtype` generates nice `str` representation, use it.
        return cls._parse_str(str(dtype))
