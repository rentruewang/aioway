# Copyright (c) AIoWay Authors - All Rights Reserved

"Tests for the exported fake helpers."

from collections import abc as cabc

import pytest
import tensordict as td
import torch
from torch._subclasses import fake_tensor as ft

from aioway.t import (
    all_fake,
    all_real,
    clone_fake,
    fake_mode,
    has_fake,
    has_real,
    is_fake_tensor,
    is_real_tensor,
    parse_attr,
    to_fake,
)


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

    assert has_fake(out)
    assert out.shape == (2, 3)
    assert out.dtype == torch.float32


def test_fake_tensor_identity(fake: torch.Tensor):
    "An already fake tensor not modified."

    assert to_fake(fake) is fake


def test_to_fake_mapping():
    out = to_fake({"a": torch.ones(2), "b": torch.zeros(3)})

    assert isinstance(out, cabc.Mapping)
    assert out.keys() == {"a", "b"}
    assert all(has_fake(val) for val in out.values())


def test_to_fake_sequence():
    out = to_fake([torch.ones(2), torch.zeros(3)])

    assert isinstance(out, list)
    assert all(has_fake(val) for val in out)


def test_to_fake_recursive():
    out = to_fake({"xs": [torch.ones(2)]})

    assert isinstance(out["xs"][0], ft.FakeTensor)


def test_to_fake_tdict():
    "Same `TensorDict`, but fake entry."

    out = to_fake(td.TensorDict({"a": torch.ones(2, 3)}, batch_size=[]))

    assert isinstance(out, td.TensorDict)
    assert has_fake(out)
    assert has_fake(out["a"])


def test_to_fake_tcls():
    "Same class, but fake field."

    out = to_fake(Point(x=torch.ones(2), y=torch.zeros(2)))

    assert isinstance(out, Point)
    assert has_fake(out)
    assert has_fake(out.x)
    assert has_fake(out.y)


@pytest.mark.parametrize("value", [1, 1.5, None, object()])
def test_real_obj_same(value):
    assert to_fake(value) == value


def test_to_fake_keeps_non_tensor_leaves():
    "Only tensors are converted, other leaves pass through."

    out = to_fake({"t": torch.ones(2), "n": 1, "s": "hi"})

    assert is_fake_tensor(out["t"])
    assert out["n"] == 1
    assert out["s"] == "hi"


def test_is_fake_tensor(fake):
    assert has_fake(fake)
    assert all_real(torch.ones(2, 3))


def test_tensor_guards(fake):
    real = torch.ones(2, 3)

    assert is_fake_tensor(fake)
    assert not is_real_tensor(fake)

    assert is_real_tensor(real)
    assert not is_fake_tensor(real)


@pytest.mark.parametrize("value", [1, "a", None, [torch.ones(2)]])
def test_tensor_guards_non_tensor(value):
    "Non-tensors (including containers of tensors) are neither real nor fake tensors."

    assert not is_fake_tensor(value)
    assert not is_real_tensor(value)


def test_any_fake_container(fake):
    "A container is fake if one value is fake."

    assert has_fake([torch.ones(2), fake])
    assert has_fake({"real": torch.ones(2), "fake": fake})
    assert not has_fake([torch.ones(2), torch.zeros(3)])


def test_all_fake(fake):
    assert all_fake(fake)
    assert all_fake([fake, fake])
    assert all_fake({"a": fake, "b": [fake]})

    assert not all_fake(torch.ones(2))
    assert not all_fake([fake, torch.ones(2)])


def test_has_real(fake):
    assert has_real(torch.ones(2))
    assert has_real([fake, torch.ones(2)])

    assert not has_real(fake)
    assert not has_real([fake, fake])


def test_all_real_mixed(fake):
    assert all_real([torch.ones(2), torch.zeros(3)])
    assert not all_real([torch.ones(2), fake])


def test_non_tensor_leaves_ignored(fake):
    "Predicates only look at tensors."

    assert all_fake([fake, 1, "a", None])
    assert all_real([torch.ones(2), 1, "a", None])


@pytest.mark.parametrize("empty", [[], {}, ()])
def test_empty_container_real(empty):
    assert all_real(empty)


@pytest.mark.parametrize("empty", [[], {}, ()])
def test_empty_container_fake(empty):
    "No tensors means vacuously all fake, and nothing real."

    assert all_fake(empty)
    assert not has_real(empty)


def test_is_fake_tdict(fake):
    assert has_fake(td.TensorDict({"a": fake}, batch_size=[]))
    assert not has_fake(td.TensorDict({"a": torch.ones(2, 3)}, batch_size=[]))


def test_all_fake_tdict(fake):
    real = td.TensorDict({"a": torch.ones(2, 3)}, batch_size=[])
    faked = td.TensorDict({"a": fake}, batch_size=[])

    assert all_fake(faked) and not has_real(faked)
    assert all_real(real) and has_real(real)

    mixed = {"real": real, "fake": faked}
    assert has_fake(mixed) and has_real(mixed)
    assert not all_fake(mixed) and not all_real(mixed)


def test_is_fake_tcls(fake):
    assert has_fake(Point(x=fake, y=fake))
    assert not has_fake(Point(x=torch.ones(2), y=torch.ones(2)))


def test_all_fake_tcls(fake):
    assert all_fake(Point(x=fake, y=fake))
    assert not has_real(Point(x=fake, y=fake))
    assert has_real(Point(x=torch.ones(2), y=torch.ones(2)))


def test_clone_fake_tensor(fake):
    assert has_fake(fake)

    out = clone_fake(fake)

    assert has_fake(out)
    assert out is not fake

    assert parse_attr(out) == parse_attr(fake)


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
    assert has_fake(out[0])


def test_clone_fake_tdict(fake):
    tdict = td.TensorDict({"a": fake}, batch_size=[])

    out = clone_fake(tdict)

    assert out is not tdict
    assert has_fake(out)


def test_clone_fake_tdict_entries(fake):
    "Entries get new ids, same shape."

    out = clone_fake(td.TensorDict({"a": fake}, batch_size=[]))

    assert isinstance(out, td.TensorDict)
    assert out["a"] is not fake
    assert all_fake(out)
    assert parse_attr(out["a"]) == parse_attr(fake)


def test_clone_fake_tcls(fake):
    point = Point(x=fake, y=fake)

    out = clone_fake(point)

    assert isinstance(out, Point)
    assert out.x is not fake
    assert out.y is not fake
    assert all_fake(out)
