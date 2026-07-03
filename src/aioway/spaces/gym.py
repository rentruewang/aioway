# Copyright (c) AIoWay Authors - All Rights Reserved

import torch

from aioway.spaces import DType, Shape

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

    def contains(self, value: torch.Tensor) -> bool:
        return (
            value.ndim == 0
            and value.dtype
            in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            )
            and 0 <= value.item() < self.n
        )


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

    def contains(self, value: torch.Tensor) -> bool:
        if value.shape != self.low.shape:
            return False

        return bool(torch.all((value >= self.low) & (value <= self.high)).item())


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

    def contains(self, value: torch.Tensor) -> bool:
        if value.shape != self.nvec.shape:
            return False

        if DType.parse(value).family not in ["uint", "int"]:
            return False

        return bool(torch.all((value >= 0) & (value < self.nvec)).item())


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

    def contains(self, value: torch.Tensor) -> bool:
        if value.shape != self.shape:
            return False

        return bool(torch.all((value == 0) | (value == 1)).item())
