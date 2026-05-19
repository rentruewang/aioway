# Copyright (c) AIoWay Authors - All Rights Reserved

import abc

from torch import nn

from aioway._keyed import Keyed
from aioway._types import dcls_no_repr

__all__ = ["MessFwd"]


@dcls_no_repr
class MessFwd(Keyed[type[nn.Module]], abc.ABC):
    """
    `MessFwd` contains info about how a module's runtime signature looks like.
    """
