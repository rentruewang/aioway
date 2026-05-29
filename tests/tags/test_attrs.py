# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway.attrs import Attr
from aioway.tags import extract_tags, tag_attr


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
    assert len(extract_tags(tensor)) == 0
    tag_attr(attr=attr_ok, item=tensor)
    assert len(extract_tags(tensor)) == 5


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
    assert len(extract_tags(tensor)) == 0
    with pytest.raises(ValueError):
        tag_attr(attr_not_ok, tensor)
