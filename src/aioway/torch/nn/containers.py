# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn

from aioway.modes import NnInitThunk

from .modules import NnInit_, nn_init_dcls
from .ufuncs import NnLayerUFunc

__all__ = ["Sequential"]


@nn_init_dcls
class Sequential(NnInit_):
    """
    The wrapper for `nn.Sequential`.

    Since the API of `nn.Sequential` takes in `*nn.Module`s,
    and there is no easy way of making that into a dataclass,
    we must overwrite `.__call__` because the default is `NN(**dcls.asdict(self))`.
    """

    NN = nn.Sequential
    UFUNC = NnLayerUFunc

    modules: tuple[nn.Module, ...]
    """
    A list of already initialized `nn.Module` objects.
    """

    def __init__(self, *args: nn.Module):
        super().__init__()
        self.modules = args

    @typing.override
    def init_nn(self) -> nn.Module:
        # Create `nn.Sequential` instance with `NnInitThunk` is the best way
        # to ensure that the modes are invoked properly.
        thunk = NnInitThunk(self.NN, *self.modules)
        return thunk()
