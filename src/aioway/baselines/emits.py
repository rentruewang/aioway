# Copyright (c) AIoWay Authors - All Rights Reserved

"`Emitter`s generate the simplest unoptimized solutions, based on constraints."

from collections import abc as cabc

from aioway._iters import UFunc
from aioway.spaces import Space

from .baselines import registered_baselines

__all__ = ["emit"]


def emit(observation_space: Space, action_space: Space) -> cabc.Generator[UFunc]:
    """
    Emit some candidates based on the given spaces.
    """

    for baseline in registered_baselines():
        result = baseline(observation_space, action_space)

        if result is not NotImplemented:
            yield result
