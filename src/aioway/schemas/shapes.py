# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import logging
import typing
from collections import abc as cabc

import numpy as np
import torch
from numpy import typing as npt

from aioway._common import get_tracker, is_list_of, is_tuple_of

__all__ = ["ShapeLike", "Shape"]

LOGGER = logging.getLogger(__name__)
TRACKER = get_tracker(lambda: Shape)

type _PrimitiveNumber = float | int | bool
type _IntArrayLike = tuple[int, ...] | list[int] | npt.NDArray[np.int_]
type ShapeCmpType = Shape | torch.Size | _IntArrayLike | _PrimitiveNumber

type ShapeLike = int | cabc.Iterable[int] | Shape
"Types convertible to `Shape`s. Note that `int` can be converted as well."


_is_tuple_of_int = is_tuple_of(int)
_is_list_of_int = is_list_of(int)


@dcls.dataclass(frozen=True, eq=False)
class Shape(cabc.Sequence[int]):
    """
    `Shape` represents a regular (non-jagged) array's dimensions,
    must be a `tuple` like object, and `tuple` would be used on it.

    Right now, it represents the shape of a `Tensor` **outside** the batch dimension.
    """

    dims: npt.NDArray[np.uint]
    """
    The dimensions
    """

    def __post_init__(self):

        if not self.valid():
            raise ValueError(self)

        self.dims.flags.writeable = False

        LOGGER.debug("Shape created: %s", self)

    def __hash__(self):
        return hash(self.dims.tobytes())

    @typing.override
    def __repr__(self) -> str:
        return "(" + "x".join(map(str, self.dims)) + ")"

    def __eq__(self, other: object) -> bool:
        if (rhs := _as_int_arr(other)) is None:
            return NotImplemented

        lhs = np.asarray(self)
        return lhs.ndim == rhs.ndim and (lhs == rhs).all().item()

    def exceeds(self, other: typing.Self):
        if self.ndim != other.ndim:
            raise ValueError

        lhs = np.asarray(self)
        rhs = np.asarray(other)

        return (lhs > rhs).any().item()

    @typing.override
    def __len__(self):
        return len(self.dims)

    @typing.overload
    def __getitem__(self, idx: int) -> int: ...

    @typing.overload
    def __getitem__(self, idx: slice) -> typing.Self: ...

    @typing.override
    def __getitem__(self, idx):
        match idx:
            case int():
                return self.dims[idx]
            case slice():
                return type(self)(self.dims[idx])
            case _:
                raise TypeError(type(idx))

    @typing.override
    def __iter__(self) -> cabc.Iterator[int]:
        return iter(self.dims)

    def __array__(self):
        return self.dims

    def unsqueeze(self, dim: int):
        dims = list(self.dims)
        dims.insert(dim, 1)
        return self.parse(dims)

    def concrete(self) -> tuple[int, ...]:
        """
        Since `Shape` may have negative dimensions, this generates a valid dimension.
        """
        return tuple(self._concrete())

    def _concrete(self):
        for i in self:
            if i < 0:
                yield 1
            else:
                yield i

    def valid(self) -> bool:
        """
        Check if the shape is a valid `Shape`.
        """

        LOGGER.debug("Checking if %s is a `Shape`", self)

        return np.isdtype(self.dims.dtype, "unsigned integer")

    def valid_dims(self, dims: list[int]) -> bool:
        """
        Validate the dimensions amongst the shapes.
        """

        is_list_of_int = is_list_of(int)
        length = len(self)

        return is_list_of_int(dims) and (max(dims) >= length or min(dims) < -length)

    def wrap_index(self, dims: list[int]) -> list[int]:
        # Make the dims positive.
        return [d % len(self) for d in dims]

    def numel(self) -> int:
        return np.prod(self.dims).item()

    def broadcastable(self, other: ShapeLike):
        try:
            _ = self.broadcast(other)
            return True
        except ValueError:
            return False

    def broadcast(self, other: ShapeLike):
        other = Shape.parse(other)

        try:
            result = np.broadcast_shapes(self.dims, other.dims)
        except ValueError as ve:
            raise ValueError(f"{self=} and {other=} cannot be broadcasted together.")

        return Shape.parse(*result)

    @property
    def ndim(self) -> int:
        """
        Number of dimensions in a shape.
        """
        return len(self)

    @property
    def size(self) -> int:
        """
        Number of elements in a shape.
        """

        total = 1

        for s in self:
            total *= s

        return total

    @typing.overload
    @classmethod
    def parse(cls, *dims: int) -> typing.Self: ...

    @typing.overload
    @classmethod
    def parse(cls, dim: ShapeLike, /) -> typing.Self: ...

    @classmethod
    def parse(cls, *dims) -> typing.Self:
        """
        Convenience constructor for `Shape`.

        Takes either of the following signature:

        1. `Shape.parse(*dims)`. Here dims must be integers.
        2. `Shape.parse(iterable)`. Here dims must be iterable. No additional args.
        3. `Shape.parse(Shape)`. Returns it by reference.
        """

        try:
            # `Shape.parse(*int)`.
            if _is_tuple_of_int(dims):
                return cls._shape(dims)

            # `shape(iterable)`.
            elif len(dims) == 1:
                return cls._shape(dims[0])

            raise ValueError
        except ValueError:
            raise ValueError(*dims)

    @classmethod
    def _shape(cls, dims) -> typing.Self:
        "Try converting dims to `Shape`, raise `ValueError` on failure."

        if isinstance(dims, cls):
            return dims

        if isinstance(dims, cabc.Iterable):
            dims_array = tuple(dims)

            if _is_tuple_of_int(dims_array):
                return cls(np.array(dims_array).astype("uint"))

        raise ValueError


def _as_int_arr(obj: object) -> npt.NDArray[np.uint] | None:
    """
    Convert the object into a numpy array. Return `None` if it's not doable / not integral array.
    """

    array = np.asarray(obj)

    if not np.isdtype(array.dtype, "integral"):
        return None

    return array.astype("uint")
