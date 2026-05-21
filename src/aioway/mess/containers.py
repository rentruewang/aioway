# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn

from aioway.modes import NnInitFn

from .fwds import InputFwd
from .mess import Mess, MessInit, mess_init_dcls

__all__ = []


@mess_init_dcls
class SequentialInit(MessInit):
    """
    The wrapper for `nn.Sequential`.

    Since the API of `nn.Sequential` takes in `*nn.Module`s,
    and there is no easy way of making that into a dataclass,
    we must overwrite `.do` because the default is `KEY(**dcls.asdict(self))`.
    """

    modules: tuple[nn.Module, ...]
    """
    A list of already initialized `nn.Module` objects.
    """

    def __init__(self, *args: nn.Module):
        super().__init__()
        self.modules = args

    @typing.override
    def init(self, module: type[nn.Module]) -> nn.Module:
        # Create `nn.Sequential` instance with `NnInitFn` is the best way
        # to ensure that the modes are invoked properly.
        return NnInitFn(func=module, args=self.modules, kwargs={}).do()


_ = Mess(nn_type=nn.Sequential, init=SequentialInit, fwd=InputFwd)
