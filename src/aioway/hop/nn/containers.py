# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn

from aioway.modes import NnInitFn

from .modules import NnInit, nn_init_dcls

__all__ = ["Sequential"]


@nn_init_dcls
class Sequential(NnInit):
    """
    The wrapper for `nn.Sequential`.

    Since the API of `nn.Sequential` takes in `*nn.Module`s,
    and there is no easy way of making that into a dataclass,
    we must overwrite `.__call__` because the default is `NN(**dcls.asdict(self))`.
    """

    NN = nn.Sequential

    modules: tuple[nn.Module, ...]
    """
    A list of already initialized `nn.Module` objects.
    """

    def __init__(self, *args: nn.Module):
        super().__init__()
        self.modules = args

    @typing.override
    def init_nn(self) -> nn.Module:
        # Create `nn.Sequential` instance with `NnInitFn` is the best way
        # to ensure that the modes are invoked properly.
        thunk = NnInitFn(func=self.NN, args=self.modules, kwargs={})
        return thunk()
