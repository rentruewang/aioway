# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway.cc import LocalScope, VarInfo, VarList
from aioway.t import fake_mode


def _make_fake() -> torch.Tensor:
    with fake_mode():
        return torch.zeros(3)


def make_real() -> torch.Tensor:
    return torch.zeros(3)


@pytest.fixture
def x_info() -> VarInfo:
    info = VarInfo(producer=-1, fake=_make_fake())
    info.add_consumers(0, 1)
    return info


@pytest.fixture
def y_info() -> VarInfo:
    info = VarInfo(producer=0, fake=_make_fake())
    info.add_consumers(1)
    return info


@pytest.fixture
def var_list(x_info, y_info) -> VarList:
    return VarList([x_info, y_info])


@pytest.fixture
def var_scope(var_list: VarList) -> LocalScope:
    return LocalScope(var_list)


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


def test_len_locals(var_scope: LocalScope):
    assert len(var_scope) == 2


def test_no_dup_vars(x_info):
    with pytest.raises(ValueError):
        LocalScope(VarList([x_info, x_info]))


def test_update_to_real(var_scope, x_info):
    real = make_real()
    var_scope.update(x_info.fake, real)

    assert var_scope[x_info.fake] is real
    assert var_scope.value(x_info.fake) is real


def test_update_to_real_containers(var_scope, x_info, y_info):
    x, y = make_real(), make_real()
    var_scope.update({"x": x_info.fake, "ys": [y_info.fake]}, {"x": x, "ys": [y]})

    assert var_scope[x_info.fake] is x
    assert var_scope[y_info.fake] is y


def test_update_not_same_structure(var_scope, x_info, y_info):
    with pytest.raises(ValueError):
        var_scope.update([x_info.fake, y_info.fake], [make_real()])


def test_update_keeps_real(var_scope: LocalScope, x_info):
    const, real = make_real(), make_real()
    var_scope.update([x_info.fake, const], [real, const])

    assert var_scope[x_info.fake] is real
    assert const not in var_scope


def test_getitem_be_fake(var_scope):
    with pytest.raises(KeyError):
        var_scope[make_real()]


def test_getitem_missing_fake(var_scope: LocalScope, x_info):
    assert var_scope[x_info.fake] is None


def test_expire_free(var_scope: LocalScope, x_info: VarInfo, y_info: VarInfo):
    assert x_info.consumers == {0, 1}
    assert y_info.consumers == {1}
    var_scope.update([x_info.fake, y_info.fake], [make_real(), make_real()])

    var_scope.expire(0)
    assert var_scope.is_alive(x_info)
    assert var_scope.is_alive(y_info)

    var_scope.expire(1)
    assert not var_scope.is_alive(x_info)
    assert not var_scope.is_alive(y_info)


def test_map_to_real(var_scope: LocalScope, x_info: VarInfo, y_info: VarInfo):
    x, y = make_real(), make_real()
    var_scope.update([x_info.fake, y_info.fake], [x, y])

    mapped = var_scope.map({"x": x_info.fake, "ys": [y_info.fake]})

    assert mapped["x"] is x
    assert mapped["ys"][0] is y


def test_map_keeps_real(var_scope: LocalScope, x_info: VarInfo):
    real, const = make_real(), make_real()
    var_scope.update(x_info.fake, real)

    mapped = var_scope.map([x_info.fake, const])

    assert mapped[0] is real
    assert mapped[1] is const


def test_map_keeps_non_tensors(var_scope: LocalScope, x_info: VarInfo):
    real = make_real()
    var_scope.update(x_info.fake, real)

    mapped = var_scope.map({"t": x_info.fake, "n": 3, "s": "hi"})

    assert mapped["t"] is real
    assert mapped["n"] == 3
    assert mapped["s"] == "hi"


def test_map_allow_missing(var_scope: LocalScope, x_info: VarInfo):
    var_scope.map([x_info.fake])


def test_var_list_len(var_list: VarList):
    assert len(var_list) == 2


def test_var_list_no_dup(x_info: VarInfo):
    with pytest.raises(ValueError):
        VarList([x_info, x_info])


def test_var_list_contains(var_list: VarList, x_info: VarInfo, y_info: VarInfo):
    assert x_info.fake in var_list
    assert id(y_info.fake) in var_list
    assert _make_fake() not in var_list


def test_var_list_getitem(var_list: VarList, x_info: VarInfo, y_info: VarInfo):
    assert var_list[x_info.fake] is x_info
    assert var_list[id(y_info.fake)] is y_info


def test_var_list_iter(var_list: VarList, x_info: VarInfo, y_info: VarInfo):
    assert set(var_list) == {id(x_info.fake), id(y_info.fake)}


def test_var_list_values(var_list: VarList, x_info: VarInfo, y_info: VarInfo):
    assert list(var_list.values()) == [x_info, y_info]


def test_var_list_producers(var_list: VarList, x_info: VarInfo, y_info: VarInfo):
    assert list(var_list.producers(-1)) == [x_info]
    assert list(var_list.producers(0)) == [y_info]


def test_var_list_consumers(var_list: VarList, x_info: VarInfo, y_info: VarInfo):
    assert list(var_list.consumers(0)) == [x_info]
    assert list(var_list.consumers(1)) == [x_info, y_info]
