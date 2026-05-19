# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn

from aioway._keyed import Keyed

__all__ = ["MessFwd"]


class MessFwd(Keyed[type[nn.Module]]):
    pass
