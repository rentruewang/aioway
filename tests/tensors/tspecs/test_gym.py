# Copyright (c) AIoWay Authors - All Rights Reserved

import numpy as np
import pytest
import torch
from torchrl.data import tensor_specs as tspecs


@pytest.fixture(autouse=True)
def import_gym_or_skip():
    try:
        import gymnasium
    except ImportError:
        raise pytest.skip("No gymnasium")
    else:
        return gymnasium


@pytest.fixture
def gym(import_gym_or_skip):
    return import_gym_or_skip


@pytest.fixture
def gs(gym):
    return gym.spaces


@pytest.fixture
def gym_space_tspec():
    from aioway.tensors import gym_space_tspec

    return gym_space_tspec


def test_gym_unbounded(gs, gym_space_tspec):
    space = gs.Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float32)
    tspec = gym_space_tspec(space)

    assert isinstance(tspec, tspecs.Bounded)
    assert torch.all(tspec.low == -1)
    assert torch.all(tspec.high == 2)
    assert tspec.shape == (3,)
    assert tspec.dtype == torch.float32


def test_gym_box_element(gs, gym_space_tspec):
    low = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    high = np.array([1.0, 0.0, 10.0], dtype=np.float32)
    tspec = gym_space_tspec(gs.Box(low=low, high=high, dtype=np.float32))

    assert torch.all(tspec.low == torch.from_numpy(low))
    assert torch.all(tspec.high == torch.from_numpy(high))


def test_box_inf_both_sides(gs, gym_space_tspec):
    space = gs.Box(low=-np.inf, high=np.inf, shape=(2, 2), dtype=np.float32)
    tspec = gym_space_tspec(space)

    assert isinstance(tspec, tspecs.Unbounded)
    assert tspec.shape == (2, 2)
    assert tspec.dtype == torch.float32


def test_box_inf_one_side(gs, gym_space_tspec):
    space = gs.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)
    tspec = gym_space_tspec(space)

    assert isinstance(tspec, tspecs.Bounded)
    assert not isinstance(tspec, tspecs.Unbounded)
    assert torch.isinf(tspec.high).all()


def test_discrete_to_categorical(gs, gym_space_tspec):
    tspec = gym_space_tspec(gs.Discrete(5))

    assert isinstance(tspec, tspecs.Categorical)
    assert tspec.n == 5
    assert tspec.shape == torch.Size(())
    assert tspec.dtype is torch.int64


def test_multi_discrete_multi_categorical(gs, gym_space_tspec):
    tspec = gym_space_tspec(gs.MultiDiscrete([3, 2, 4]))

    assert isinstance(tspec, tspecs.MultiCategorical)
    assert tspec.shape == torch.Size((3,))
    assert tspec.nvec.tolist() == [3, 2, 4]


def test_dict_maps_to_composite(gs, gym_space_tspec):
    space = gs.Dict(
        {
            "obs": gs.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "id": gs.Discrete(3),
        }
    )
    tspec = gym_space_tspec(space)

    assert isinstance(tspec, tspecs.Composite)
    assert tspec.shape == torch.Size(())
    assert set(tspec.keys()) == {"obs", "id"}
    assert isinstance(tspec["obs"], tspecs.Bounded)
    assert tspec["obs"].shape == torch.Size((2,))
    assert isinstance(tspec["id"], tspecs.Categorical)
