# Copyright (c) AIoWay Authors - All Rights Reserved

"`Vector` is a homogenious array of values."

import logging
import operator
import typing

import torch

from aioway._common import AnyUFunc2
from aioway.fn import TensorFn
from aioway.schemas import Attr, attr

__all__ = ["Vector"]

LOGGER = logging.getLogger(__name__)


type VectorRhs = Vector | torch.Tensor | int | float | bool


class Vector:
    """
    A `Vector` is a `torch.Tensor` plus its `meta.Attr`.
    """

    __match_args__ = "data", "attr"

    def __init__(self, data: torch.Tensor) -> None:
        self._data = data
        "The underlying data."

    def __repr__(self):
        return f"Vector({self.attr})"

    def __neg__(self) -> typing.Self:
        return self.__op1(operator.neg)

    def __invert__(self) -> typing.Self:
        return self.__op1(operator.invert)

    def __add__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.add)

    def __sub__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.sub)

    def __mul__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.mul)

    def __truediv__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.truediv)

    def __floordiv__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.floordiv)

    def __mod__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.mod)

    def __pow__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.pow)

    @typing.no_type_check
    def __eq__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.eq)

    @typing.no_type_check
    def __ne__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.ne)

    @typing.no_type_check
    def __ge__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.ge)

    @typing.no_type_check
    def __gt__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.gt)

    @typing.no_type_check
    def __le__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.le)

    @typing.no_type_check
    def __lt__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.lt)

    def __op1(self, unop: typing.Any) -> typing.Self:
        return type(self)(unop(self.data))

    def __op2(self, other: typing.Any, binop: AnyUFunc2) -> typing.Self:
        match other:
            case Vector():
                return type(self)(binop(self.data, other.data))
            case TensorFn():
                raise NotImplementedError
            case int() | float() | bool():
                return type(self)(binop(self.data, other))

        raise NotImplementedError

    def torch(self) -> torch.Tensor:
        "Get the `torch.Tensor` data that this `Vector` contains."
        return self._data

    def typeof(self) -> Attr:
        "Get the type information `meta.Attr` of the `torch.Tensor` that this `Vector` represents."
        return self.attr

    def cpu(self) -> typing.Self:
        data = self._data.cpu()
        return type(self)(data=data)

    def numpy(self):
        return self.torch().cpu().numpy()

    def tolist(self):
        return self.torch().tolist()

    def fn(self):
        return TensorFn.from_tensor(self.data)

    @property
    def data(self):
        return self._data

    @property
    def attr(self):
        return attr(self.data)
