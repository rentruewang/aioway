# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from aioway.hop import HopInit
from aioway.spaces import Space


class Builder(typing.Protocol):
    def __call__(self, inputs: list[Space], outputs: list[Space]) -> list[HopInit]:
        """
        Compiles from
        """

        ...
