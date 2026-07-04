# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import torch

from aioway.attrs import Attr, DType, Shape
from aioway.errors import re_raise_func

from .spaces import TensorSpace, space_dcls

__all__ = ["DiscreteSpace", "BoxSpace", "MultiDiscreteSpace", "MultiBinarySpace"]


@space_dcls
class DiscreteSpace(TensorSpace):
    """
    A scalar integer in the range ``[0, n)``.
    """

    n: int
    "The number of discrete values."

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("'n' must be positive.")

    @typing.override
    @re_raise_func(AssertionError, ValueError)
    def _check_attr(self, attr: Attr) -> None:
        assert attr.ndim == 1
        assert attr.dtype.family in ["uint", "int"]

    @typing.override
    @re_raise_func(AssertionError, ValueError)
    def _check_data(self, value: torch.Tensor) -> None:
        assert torch.all((0 <= value) & (value <= self.n)).item()


@space_dcls
class BoxSpace(TensorSpace):
    """
    A continuous tensor bounded elementwise by ``low`` and ``high``.
    """

    low: torch.Tensor
    "The inclusive lower bound."

    high: torch.Tensor
    "The inclusive upper bound."

    def __post_init__(self) -> None:

        def check[T](name: str):
            if getattr(self.low, name) != getattr(self.high, name):
                raise ValueError(f"'low' and 'high' must have the same {name}.")

        check("shape")
        check("dtype")
        check("device")

        if not torch.all(self.low <= self.high):
            raise ValueError("'low' must be less than or equal to 'high' elementwise.")

    @re_raise_func(AssertionError, ValueError)
    def _check_attr(self, attr: Attr) -> None:
        assert attr.ndim == self.low.ndim + 1
        assert attr.shape[1:] == self.low.shape
        assert attr.dtype.family == "float"

    @re_raise_func(AssertionError, ValueError)
    def _check_data(self, value: torch.Tensor) -> None:
        assert torch.all((value >= self.low) & (value <= self.high)).item()


@space_dcls
class MultiDiscreteSpace(TensorSpace):
    """
    A tensor of independent discrete values.
    """

    nvec: torch.Tensor
    "The exclusive upper bound for each element."

    def __post_init__(self) -> None:
        if self.nvec.ndim == 0:
            raise ValueError("'nvec' must have at least one dimension.")

        if DType.parse(self.nvec.dtype).family not in ["uint", "int"]:
            raise TypeError("'nvec' must have an integer dtype.")

        if not torch.all(self.nvec > 0):
            raise ValueError("All elements of 'nvec' must be positive.")

    @typing.override
    @re_raise_func(AssertionError, ValueError)
    def _check_attr(self, attr: Attr) -> None:
        assert attr.ndim == self.nvec.ndim + 1
        assert attr.shape[1:] == self.nvec.shape
        assert attr.dtype.family in ["uint", "int"]

    @typing.override
    def _check_data(self, value: torch.Tensor) -> None:
        nvec_expanded = self.nvec[None, ...]
        assert torch.all((value >= 0) & (value < nvec_expanded)).item()


@space_dcls
class MultiBinarySpace(TensorSpace):
    """
    A tensor whose elements are either 0 or 1.
    """

    shape: Shape
    "The expected tensor shape."

    def __post_init__(self) -> None:
        if len(self.shape) == 0:
            raise ValueError("'shape' must not be empty.")

        if any(d <= 0 for d in self.shape):
            raise ValueError("All dimensions of 'shape' must be positive.")

    @typing.override
    @re_raise_func(AssertionError, ValueError)
    def _check_attr(self, attr: Attr) -> None:
        assert attr.ndim == self.shape.ndim + 1
        assert attr.shape[1:] == self.shape
        assert attr.dtype.family in ["bool", "int", "uint"]

    @typing.override
    @re_raise_func(AssertionError, ValueError)
    def _check_data(self, value: torch.Tensor) -> None:
        assert torch.all((value == 0) | (value == 1)).item()
