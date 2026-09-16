# Copyright (c) AIoWay Authors - All Rights Reserved

"Tests for the exported fake helpers: `to_fake`, `is_fake`, `is_real`, `clone_fake`."

from collections import abc as cabc

import pytest
import tensordict as td
import torch
from torch._subclasses import fake_tensor as ft

from aioway.tensors import Attr, clone_fake, fake_mode, is_fake, is_real, to_fake


@td.tensorclass
class Point:
    "A minimal tensorclass, used to exercise the tensorclass branch."

    x: torch.Tensor
    y: torch.Tensor


@pytest.fixture
def fake() -> torch.Tensor:
    "A fake tensor built straight from `fake_mode`, so it doesn't depend on `to_fake`."

    with fake_mode():
        return torch.ones(2, 3)


def test_to_fake_tensor():
    "A real tensor becomes a `FakeTensor` keeping its shape and dtype."

    out = to_fake(torch.ones(2, 3))

    assert is_fake(out)
    assert out.shape == (2, 3)
    assert out.dtype == torch.float32


def test_fake_tensor_identity(fake: torch.Tensor):
    "An already fake tensor not modified."

    assert to_fake(fake) is fake


def test_to_fake_mapping():
    out = to_fake({"a": torch.ones(2), "b": torch.zeros(3)})

    assert isinstance(out, cabc.Mapping)
    assert out.keys() == {"a", "b"}
    assert all(is_fake(val) for val in out.values())


def test_to_fake_sequence():
    out = to_fake([torch.ones(2), torch.zeros(3)])

    assert isinstance(out, list)
    assert all(is_fake(val) for val in out)


def test_to_fake_recursive():
    out = to_fake({"xs": [torch.ones(2)]})

    assert isinstance(out["xs"][0], ft.FakeTensor)


def test_to_fake_tdict():
    "Same `TensorDict`, but fake entry."

    out = to_fake(td.TensorDict({"a": torch.ones(2, 3)}, batch_size=[]))

    assert isinstance(out, td.TensorDict)
    assert is_fake(out)
    assert is_fake(out["a"])


def test_to_fake_tcls():
    "Same class, but fake field."

    out = to_fake(Point(x=torch.ones(2), y=torch.zeros(2)))

    assert isinstance(out, Point)
    assert is_fake(out)
    assert is_fake(out.x)
    assert is_fake(out.y)


@pytest.mark.parametrize("value", [1, 1.5, None, object()])
def test_real_obj_same(value):
    assert to_fake(value) == value


def test_is_fake_tensor(fake):
    assert is_fake(fake)
    assert is_real(torch.ones(2, 3))


def test_any_fake_container(fake):
    "A container is fake if one value is fake."

    assert is_fake([torch.ones(2), fake])
    assert is_fake({"real": torch.ones(2), "fake": fake})
    assert not is_fake([torch.ones(2), torch.zeros(3)])


@pytest.mark.parametrize("empty", [[], {}, ()])
def test_empty_container_real(empty):
    assert is_real(empty)


def test_is_fake_tdict(fake):
    assert is_fake(td.TensorDict({"a": fake}, batch_size=[]))
    assert not is_fake(td.TensorDict({"a": torch.ones(2, 3)}, batch_size=[]))


def test_is_fake_tcls(fake):
    assert is_fake(Point(x=fake, y=fake))
    assert not is_fake(Point(x=torch.ones(2), y=torch.ones(2)))


def test_clone_fake_tensor(fake):
    assert is_fake(fake)

    out = clone_fake(fake)

    assert is_fake(out)
    assert out is not fake

    assert Attr.from_tensor(out) == Attr.from_tensor(fake)


def test_clone_fake_real_same():
    real = torch.ones(2, 3)

    assert clone_fake(real) is real


def test_clone_fake_mapping(fake):
    real = torch.ones(2, 3)

    out = clone_fake({"fake": fake, "real": real})

    assert out.keys() == {"fake", "real"}
    assert out["fake"] is not fake
    assert out["real"] is real


def test_clone_fake_sequence(fake):
    out = clone_fake([fake])

    assert isinstance(out, list)
    assert out[0] is not fake
    assert is_fake(out[0])


def test_clone_fake_tdict(fake):
    tdict = td.TensorDict({"a": fake}, batch_size=[])

    out = clone_fake(tdict)

    assert out is not tdict
    assert is_fake(out)
