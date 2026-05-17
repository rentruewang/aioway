# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn

from .might import Might

__all__ = ["Sequential"]


# @dcls_no_repr
class Sequential(Might):
    """
    The wrapper for `nn.Sequential`.

    There is no easy way to model
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
        from aioway.fn import NnInitFn

        # Create `nn.Sequential` instance with `NnInitFn` is the best way
        # to ensure that the modes are invoked properly.
        return NnInitFn(func=self.KEY, args=self.modules, kwargs={}).do()
