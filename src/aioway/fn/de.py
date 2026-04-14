# Copyright (c) AIoWay Authors - All Rights Reserved

"De stands for defer / eager."

import typing
from collections import abc as cabc
from typing import Any

import numpy as np
import tensordict as td
import torch
from numpy import typing as npt

from .fn import DoFn
from .tdicts import TensorDictFn, tdict
from .tensors import TensorFn, tensor

__all__ = ["defer", "eager"]


@typing.overload
def defer(value: bool, /) -> bool: ...


@typing.overload
def defer(value: int, /) -> int: ...


@typing.overload
def defer(value: float, /) -> float: ...


@typing.overload
def defer(value: slice, /) -> slice: ...


@typing.overload
def defer(value: str, /) -> str: ...


@typing.overload
def defer(value: None, /) -> None: ...


@typing.overload
def defer(value: npt.NDArray, /) -> TensorFn: ...


@typing.overload
def defer(value: torch.Tensor | TensorFn, /) -> TensorFn: ...


@typing.overload
def defer(value: td.TensorDict | TensorDictFn, /) -> TensorDictFn: ...


@typing.overload
def defer(value: cabc.Sequence[Any], /) -> TensorFn: ...


@typing.overload
def defer(value: cabc.Mapping[str, Any], /) -> TensorDictFn: ...


@typing.no_type_check
def defer(value, /):

    match value:
        case None | int() | float() | bool() | slice() | str() | DoFn():
            return value
        case torch.Tensor():
            return tensor(value)
        case td.TensorDict():
            return tdict(value)
        case np.ndarray() | cabc.Sequence():
            return defer(torch.tensor(value))
        case cabc.Mapping():
            return defer(td.TensorDict(value))

    if isinstance(value, cabc.Mapping):
        return tdict(td.TensorDict.from_any(value))

    raise TypeError(f"We do not know how to handle: {value=}, {type(value)=}")


@typing.overload
def eager(value: bool, /) -> bool: ...


@typing.overload
def eager(value: int, /) -> int: ...


@typing.overload
def eager(value: float, /) -> float: ...


@typing.overload
def eager(value: slice, /) -> slice: ...


@typing.overload
def eager(value: None, /) -> None: ...


@typing.overload
def eager(value: npt.NDArray, /) -> npt.NDArray: ...


@typing.overload
def eager(value: torch.Tensor, /) -> torch.Tensor: ...


@typing.overload
def eager[T](value: DoFn[T], /) -> T: ...


@typing.overload
def eager(value: TensorDictFn, /) -> td.TensorDict: ...


def eager(value, /):
    match value:
        case (
            None
            | int()
            | float()
            | bool()
            | slice()
            | str()
            | np.ndarray()
            | torch.Tensor()
            | td.TensorDict()
        ):
            return value
        case DoFn():
            return value.do()

    raise TypeError(f"We do not know how to handle: {value=}, {type(value)=}")
