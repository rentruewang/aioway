# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway._iters import UFunc
from aioway.spaces import AttrSpace, ShapeSpace, Space
from aioway.torch.nn import Linear

from .baselines import register_baseline


@register_baseline
def linear_shape(observation_space: Space, action_space: Space) -> UFunc:
    """
    `Linear` module from `ShapeSpace`s.
    """

    if not isinstance(observation_space, ShapeSpace):
        return NotImplemented

    if not isinstance(action_space, ShapeSpace):
        return NotImplemented

    return Linear(
        in_features=observation_space[-1], out_features=action_space[-1]
    ).ufunc


@register_baseline
def linear_from_attr(observation_space: Space, action_space: Space) -> UFunc:
    """
    `Linear` module from `AttrShape`s.
    """

    if not isinstance(observation_space, AttrSpace):
        return NotImplemented

    if not isinstance(action_space, AttrSpace):
        return NotImplemented

    if (observ_shape := observation_space.shape) is None:
        return NotImplemented
    if (act_shape := action_space.shape) is None:
        return NotImplemented

    return Linear(in_features=observ_shape[-1], out_features=act_shape[-1]).ufunc
