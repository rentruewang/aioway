# Copyright (c) AIoWay Authors - All Rights Reserved


import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces as gs
from torchrl.data import tensor_specs as tspecs

from ._utils import exec_if_not_none, parse_dtype
from .tspecs import TSpec

__all__ = ["gym_space_tspec"]


def gym_space_tspec(space: gym.Space) -> TSpec:
    """
    Convert `gymnasium.Space` to `TSpec`.
    """

    dtype = exec_if_not_none(space.dtype, parse_dtype)
    shape = exec_if_not_none(space.shape, torch.Size)

    match space:
        case gs.Box():

            # A box that is not bounded.
            if not space.is_bounded("below") and not space.is_bounded("above"):
                return tspecs.Unbounded(shape=shape, dtype=dtype)

            # A box that is bounded in some way.
            return tspecs.Bounded(
                low=torch.as_tensor(space.low, dtype=dtype),
                high=torch.as_tensor(space.high, dtype=dtype),
                shape=shape,
                dtype=dtype,
            )

        case gs.Discrete(n=n, start=start):
            if start != 0:
                raise NotImplementedError(
                    f"Discrete(start={start}) has no torchrl equivalent; "
                    "Categorical is always 0-based. Shift the action in a "
                    "transform instead."
                )

            return tspecs.Categorical(n=int(n), shape=torch.Size(()), dtype=dtype)

        case gs.MultiDiscrete(nvec=nvec):
            start = space.start

            if np.any(start != 0):
                raise ValueError(
                    "MultiDiscrete with a non-zero start has no torchrl "
                    "equivalent; MultiCategorical is always 0-based."
                )

            return tspecs.MultiCategorical(
                nvec=torch.as_tensor(nvec, dtype=torch.long), shape=shape, dtype=dtype
            )

        case gs.MultiBinary():
            assert shape, space.shape
            return tspecs.Binary(n=int(shape[-1]), shape=shape, dtype=dtype)

        case gs.Dict(spaces=subspaces):
            return tspecs.Composite(
                {key: gym_space_tspec(sub) for key, sub in subspaces.items()}
            )

        case _:
            raise TypeError(f"Unknown type {type(space)=}.")
