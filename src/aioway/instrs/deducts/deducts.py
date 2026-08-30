# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections import abc as cabc

from aioway.tspecs import TSpec

__all__ = ["TSpecInfer", "TSpecInferLike", "TSpecInferCompat"]

type TSpecInferLike = TSpecInfer | TSpecInferCompat
"""
Types compatible with `TSpecInfer`.
"""


@typing.runtime_checkable
class TSpecInfer(typing.Protocol):
    """
    `TSpecInfer` converts from an input `TSpec` to another `TSpec`.

    It's the type of callables that consumes a torch object and outputs another one.
    """

    def __call__(self, tspec: TSpec, /) -> TSpec: ...


@typing.runtime_checkable
class TSpecInferCompat(typing.Protocol):
    """
    `TSpecInferCompat` can be converted to a `TSpecInfer`.
    """

    def __deduct__(self) -> TSpecInfer: ...
