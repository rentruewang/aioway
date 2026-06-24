# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

from aioway._ufuncs import UFunc
from aioway.spaces import Space

__all__ = ["BaseLine", "register_baseline", "registered_baselines"]

_BASELINES: set[BaseLine] = set()


class BaseLine(typing.Protocol):
    """
    The baselines that `Emitter` generates. If the space is not supported,
    `NotImplemented` should be returned.

    All the `BaseLine`s are registered, and iterated over during `Emitter` call.

    Right now a `BaseLine` is stateless, perhaps find a way to make it stateful?
    """

    __module__: str
    __qualname__: str
    __name__: str
    __doc__: str | None

    def __call__(self, observation_space: Space, action_space: Space, /) -> UFunc:
        raise NotImplementedError


def register_baseline[B: BaseLine](baseline: B) -> B:
    "Register a baseline into the registry."

    _BASELINES.add(baseline)
    return baseline


def registered_baselines() -> cabc.Generator[BaseLine]:
    "Get all the baselines that are currently registered."

    yield from _BASELINES
