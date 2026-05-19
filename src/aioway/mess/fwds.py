# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn

from aioway._keyed import Keyed

__all__ = ["MessFwd"]


class MessFwd(Keyed[type[nn.Module]]):
    """
    The signature of the forwarding pass.
    This is categorized in a few different categories.
    """
