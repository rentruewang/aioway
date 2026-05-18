# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn

from aioway._common import dcls_no_repr
from aioway.fn import NnInitFn

from .might import Might

__all__ = ["Sequential"]


@dcls_no_repr
class Sequential(Might):
    """
    The wrapper for `nn.Sequential`.

    Since the API of `nn.Sequential` takes in `*nn.Module`s,
    and there is no easy way of making that into a dataclass,
    we must overwrite `.do` because the default is `KEY(**dcls.asdict(self))`.
    """

    KEY = nn.Sequential

    modules: tuple[nn.Module, ...]
    """
    A list of `Might` objects that can be resolved later.
    """

    def __init__(self, *args: nn.Module):
        super().__init__()
        self.modules = args

    @typing.override
    def do(self) -> nn.Module:
        # Create `nn.Sequential` instance with `NnInitFn` is the best way
        # to ensure that the modes are invoked properly.
        return NnInitFn(func=self.KEY, args=self.modules, kwargs={}).do()
