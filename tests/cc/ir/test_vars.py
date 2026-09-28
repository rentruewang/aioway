# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway.cc import LocalVars, VarInfo
from aioway.torch import fake_mode


def make_fake() -> torch.Tensor:
    with fake_mode():
        return torch.zeros(3)


def make_real() -> torch.Tensor:
    return torch.zeros(3)


@pytest.fixture
def x_info() -> VarInfo:
    info = VarInfo(producer=-1, fake=make_fake())
    info.add_consumers(0, 1)
    return info


@pytest.fixture
def y_info() -> VarInfo:
    info = VarInfo(producer=0, fake=make_fake())
    info.add_consumers(1)
    return info


@pytest.fixture
def local_vars(x_info, y_info) -> LocalVars:
    return LocalVars.from_infos(x_info, y_info)


def test_no_real_tensor_allowed():
    with pytest.raises(ValueError):
        VarInfo(producer=0, fake=make_real())


def test_no_dup(y_info: VarInfo):
    with pytest.raises(IndexError):
        y_info.add_consumers(1)


def test_var_input(x_info: VarInfo, y_info: VarInfo):
    assert x_info.is_input
    assert not y_info.is_input


def test_var_alive_until(x_info: VarInfo):
    assert x_info.alive_until == 1


def test_var_tensor_life(y_info):
    real = make_real()

    assert not y_info.is_alive

    with pytest.raises(AttributeError):
        y_info.tensor

    y_info.tensor = real
    assert y_info.is_alive
    assert y_info.tensor is real

    del y_info.tensor
    assert not y_info.is_alive

    with pytest.raises(AttributeError):
        del y_info.tensor


def test_var_tensor_real(y_info):
    with pytest.raises(ValueError):
        y_info.tensor = make_fake()


def test_len_locals(local_vars: LocalVars):
    assert len(local_vars) == 2


def test_no_dup_vars(x_info):
    with pytest.raises(ValueError):
        LocalVars.from_infos(x_info, x_info)


def test_update_to_real(local_vars, x_info):
    real = make_real()
    local_vars.update(x_info.fake, real)

    assert local_vars[x_info.fake] is real
    assert x_info.tensor is real


def test_update_to_real_containers(local_vars, x_info, y_info):
    x, y = make_real(), make_real()
    local_vars.update({"x": x_info.fake, "ys": [y_info.fake]}, {"x": x, "ys": [y]})

    assert local_vars[x_info.fake] is x
    assert local_vars[y_info.fake] is y


def test_update_not_same_structure(local_vars, x_info, y_info):
    with pytest.raises(ValueError):
        local_vars.update([x_info.fake, y_info.fake], [make_real()])


def test_update_keeps_real(local_vars: LocalVars, x_info):
    const, real = make_real(), make_real()
    local_vars.update([x_info.fake, const], [real, const])

    assert local_vars[x_info.fake] is real
    assert const not in local_vars


def test_getitem_be_fake(local_vars):
    with pytest.raises(KeyError):
        local_vars[make_real()]


def test_getitem_missing_fake(local_vars: LocalVars, x_info):
    assert local_vars[x_info.fake] is None


def test_expire_free(local_vars: LocalVars, x_info: VarInfo, y_info: VarInfo):
    assert x_info.consumers == {0, 1}
    assert y_info.consumers == {1}
    local_vars.update([x_info.fake, y_info.fake], [make_real(), make_real()])

    local_vars.expire(0)
    assert x_info.is_alive
    assert y_info.is_alive

    local_vars.expire(1)
    assert not x_info.is_alive
    assert not y_info.is_alive


def test_map_to_real(local_vars: LocalVars, x_info: VarInfo, y_info: VarInfo):
    x, y = make_real(), make_real()
    local_vars.update([x_info.fake, y_info.fake], [x, y])

    mapped = local_vars.map({"x": x_info.fake, "ys": [y_info.fake]})

    assert mapped["x"] is x
    assert mapped["ys"][0] is y


def test_map_keeps_real(local_vars: LocalVars, x_info: VarInfo):
    real, const = make_real(), make_real()
    local_vars.update(x_info.fake, real)

    mapped = local_vars.map([x_info.fake, const])

    assert mapped[0] is real
    assert mapped[1] is const


def test_map_keeps_non_tensors(local_vars: LocalVars, x_info: VarInfo):
    real = make_real()
    local_vars.update(x_info.fake, real)

    mapped = local_vars.map({"t": x_info.fake, "n": 3, "s": "hi"})

    assert mapped["t"] is real
    assert mapped["n"] == 3
    assert mapped["s"] == "hi"


def test_map_allow_missing(local_vars: LocalVars, x_info: VarInfo):
    local_vars.map([x_info.fake])
