# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway.spaces import Attr
from aioway.dsets import AttrTag, TagDict


@pytest.fixture
def tensor(maybe_fake_mode):
    return torch.randn(32, 33).double().requires_grad_()


@pytest.fixture
def attr_ok():
    return Attr.build(
        dtype="float64",
        shape=[32, 33],
        device="cpu",
        layout="strided",
        requires_grad=True,
    )


def test_tagging_ok(attr_ok, tensor):
    assert len(TagDict.extract(tensor)) == 0
    tag = AttrTag.from_attr(attr_ok)
    tag.attach(tensor)
    assert len(TagDict.extract(tensor)) == 1
    assert tag in TagDict.extract(tensor)


def _attr_not_ok():
    yield Attr.build(
        dtype="float32",
        shape=[32, 33],
        device="cpu",
        layout="strided",
        requires_grad=True,
    )


@pytest.fixture(params=_attr_not_ok())
def attr_not_ok(request: pytest.FixtureRequest):
    return request.param


def test_tagging_not_ok(attr_not_ok, tensor):
    assert len(TagDict.extract(tensor)) == 0
    tag = AttrTag.from_attr(attr_not_ok)
    with pytest.raises(ValueError):
        tag.attach(tensor)
