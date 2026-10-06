# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway.t import TList, fake_mode


def _make_fake() -> torch.Tensor:
    with fake_mode():
        return torch.zeros(3)


def _make_real() -> torch.Tensor:
    return torch.zeros(3)


@pytest.fixture
def x() -> torch.Tensor:
    return _make_fake()


@pytest.fixture
def y() -> torch.Tensor:
    return _make_fake()


@pytest.fixture
def tlist(x, y) -> TList:
    return TList.from_iterable([x, y])


def test_tlist_len(tlist: TList):
    assert len(tlist) == 2


def test_tlist_dedup(x):
    assert len(TList.from_iterable([x, x])) == 1


def test_tlist_contains(tlist: TList, x, y):
    assert x in tlist
    assert id(y) in tlist
    assert _make_fake() not in tlist
    assert "x" not in tlist


def test_tlist_getitem(tlist: TList, x, y):
    assert {id(tlist[0]), id(tlist[1])} == {id(x), id(y)}


def test_tlist_iter(tlist: TList, x, y):
    assert set(tlist) == {x, y}


def test_tlist_index(tlist: TList, x, y):
    assert tlist[tlist.index(x)] is x
    assert tlist[tlist.index(id(y))] is y


def test_tlist_index_missing(tlist: TList):
    with pytest.raises(ValueError):
        tlist.index(_make_fake())


def test_tlist_eq(x, y):
    assert TList.from_iterable([x, y]) == TList.from_iterable([y, x])
    assert hash(TList.from_iterable([x, y])) != hash(TList.from_iterable([y, x]))


def test_tlist_fake_check_all(tlist: TList):
    assert tlist.all_fake
    assert tlist.any_fake


def test_tlist_fake_check_mixed(x):
    mixed = TList.from_iterable([x, _make_real()])

    assert mixed.any_fake
    assert not mixed.all_fake
