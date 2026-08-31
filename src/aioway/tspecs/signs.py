# Copyright (c) AIoWay Authors - All Rights Reserved

"The TSpec signature type."
from aioway._utils import Sign

import dataclasses as dcls
import typing
from collections import abc as cabc

from torchrl.data import tensor_specs as tspecs

__all__ = ["TSpecSign"]


@dcls.dataclass(repr=False)
class TSpecSign(tspecs.Composite):
    """
    `TSpecSign` is just a `Composite`, but `!=` a normal `Composite`.
    """

    @classmethod
    def parse_callable(cls, function: cabc.Callable) -> typing.Self:
        sign = Sign.from_callable(function)
