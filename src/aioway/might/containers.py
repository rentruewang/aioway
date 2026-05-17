# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn

from aioway._common import dcls_frozen_no_repr

from .might import Might

__all__ = ["Sequential"]


@dcls_frozen_no_repr
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

    @typing.override
    def do(self) -> nn.Module:
        from aioway.fn import NnInitFn

        # Create `nn.Sequential` instance with `NnInitFn` is the best way
        # to ensure that the modes are invoked properly.
        return NnInitFn(func=self.KEY, args=self.modules, kwargs={}).do()

    @classmethod
    @typing.override
    def create(cls, *args: nn.Module) -> typing.Self:
        return cls(modules=args)
