# Copyright (c) AIoWay Authors - All Rights Reserved

import logging
import typing
from collections import abc as cabc

import numpy as np
import torch

from aioway._utils import is_list_of, is_tuple_of

from ._bases import TorchAttrBase

__all__ = ["ShapeLike", "Shape"]

LOGGER = logging.getLogger(__name__)


type ShapeLike = int | cabc.Iterable[int] | Shape
"Types convertible to `Shape`s. Note that `int` can be converted as well."


_is_tuple_of_int = is_tuple_of(int)
_is_list_of_int = is_list_of(int)


class Shape(TorchAttrBase[torch.Size], cabc.Sequence[int]):
    """
    `Shape` represents a regular (non-jagged) array's dimensions,
    must be a `tuple` like object, and `tuple` would be used on it.

    Right now, it represents the shape of a `Tensor` **outside** the batch dimension.
    """

    __match_args__ = ("shape",)
    TYPE = torch.Size

    def __getstate__(self) -> object:
        return tuple(self._data)

    def __hash__(self):
        return hash(tuple(self._data))

    def __str__(self) -> str:
        return "(" + "x".join(map(str, self._data)) + ")"

    @typing.no_type_check
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Shape):
            return len(self) == len(other) and all(
                l == r for l, r in zip(self, other) if l >= 0 and r >= 0
            )

        if isinstance(other, np.ndarray):
            return other.ndim == 1 and self == other.tolist()

        if (
            False
            or isinstance(other, torch.Size)
            or _is_tuple_of_int(other)
            or _is_list_of_int(other)
        ):
            return self._data == tuple(other)

        return NotImplemented

    def __len__(self) -> int:
        return len(self._data)

    @typing.overload
    def __getitem__(self, idx: int) -> int: ...

    @typing.overload
    def __getitem__(self, idx: slice) -> typing.Self: ...

    @typing.override
    def __getitem__(self, idx):
        match idx:
            case int():
                return self._data[idx]
            case slice():
                return type(self)(self._data[idx])
            case _:
                raise TypeError(type(idx))

    @typing.override
    def __iter__(self) -> cabc.Iterator[int]:
        return iter(self._data)

    def __array__(self):
        return np.array(self._data)

    def exceeds(self, other: typing.Self):
        if self.ndim != other.ndim:
            raise ValueError

        lhs = np.asarray(self)
        rhs = np.asarray(other)

        return (lhs > rhs).any().item()

    def unsqueeze(self, dim: int):
        dims = list(self._data)
        dims.insert(dim, 1)
        return self.parse(dims)

    def set_dims(self, dims: cabc.Mapping[int, int]) -> typing.Self:
        """
        Set the dimension according to the `dims` mapping.
        Raises `ValueError` if the `dims` are not 0 <= dim < len(self).
        """

        if not all(0 <= dim < len(self) for dim in dims):
            raise ValueError(f"{dims=} contains invalid dimension values for {self=}.")

        result: list[int] = []

        for idx, size in enumerate(self):
            if idx in dims:
                result.append(dims[idx])
            else:
                result.append(size)

        return self.parse(result)

    def broadcastable(self, other: ShapeLike):
        try:
            _ = self.broadcast(other)
            return True
        except ValueError:
            return False

    def broadcast(self, other: ShapeLike):
        other = Shape.parse(other)

        try:
            result = np.broadcast_shapes(self._data, other._data)
        except ValueError as ve:
            raise ValueError(f"{self=} and {other=} cannot be broadcasted together.")

        return Shape.parse(*result)

    @property
    def shape(self):
        return self._data

    @property
    def ndim(self) -> int:
        """
        Number of dimensions in a shape.
        """
        return len(self)

    def numel(self) -> int:
        """
        Number of elements in a shape.
        """

        return self._data.numel()

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

        if isinstance(dims, tuple | torch.Size):
            return cls(torch.Size(dims))

        if isinstance(dims, cabc.Iterable):
            dims_array = tuple(dims)

            if _is_tuple_of_int(dims_array):
                return cls(torch.Size(dims_array))

        raise ValueError
