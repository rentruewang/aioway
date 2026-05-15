# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Fn`s corresponding to module's."

import abc
import dataclasses as dcls
import typing

from torch import nn

from .fn import Fn

__all__ = ["NnForwardFn", "NnInitFn", "PreviewFn"]


@dcls.dataclass(frozen=True)
class NnForwardFn:
    "`NnForwardFn` represents the module calls."

    module: nn.Module
    "The module for the `Fn`."

    args: typing.Any
    "The arguments of the module."


@dcls.dataclass(frozen=True)
class _ModInitBase(Fn, abc.ABC):
    """
    The thunk that stores modules
    """


class NnInitFn(_ModInitBase):
    pass


class PreviewFn(_ModInitBase):
    pass
