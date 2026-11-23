# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing

import loguru as L
from torch import nn

from aioway.io import Dset

__all__ = ["Cmd"]

# Cmd base class ====


class Cmd(abc.ABC):
    """
    The base class for commands.

    A command can modify global state or perform expensive computation.

    It exists for computation that cannot be expressed by `Op`s,
    which only handles stateless computation on 1 batch.

    E.g. Any modification on modules / datasets.
    E.g. Expensive reads e.g. over datasets.
    """
