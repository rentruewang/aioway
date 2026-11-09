# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake

from aioway.ops import (
    ConstantInit,
    InitError,
    KaimingNormalInit,
    KaimingUniformInit,
    ModuleRebuilder,
    NoInit,
    NormalInit,
    OnesInit,
    OrthogonalInit,
    TruncNormalInit,
    UniformInit,
    XavierNormalInit,
    XavierUniformInit,
    ZerosInit,
)


class Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 4)
        self.head = nn.Linear(4, 10, bias=False)
        self.head.weight = self.emb.weight
        self.frozen = nn.Parameter(torch.randn(3), requires_grad=False)
        self.register_buffer("scale", torch.ones(4))
        self.register_buffer("tmp", torch.ones(3), persistent=False)


def _tensors(module):
    return [*module.parameters(), *module.buffers()]


@pytest.fixture
def model():
    "A model with tied weights, a frozen parameter, and both kinds of buffers."
    return Tied()


@pytest.fixture(
    params=[
        NoInit(),
        ZerosInit(),
        OnesInit(),
        ConstantInit(val=3.5),
        UniformInit(a=-0.5, b=0.5),
        NormalInit(std=0.02),
        TruncNormalInit(a=-0.1, b=0.1),
    ],
    ids=repr,
)
def init(request):
    "Inits that work on tensors of any shape, so they can run on `model`."
    return request.param


@pytest.fixture(
    params=[
        XavierUniformInit(),
        XavierNormalInit(),
        KaimingUniformInit(),
        KaimingNormalInit(),
        OrthogonalInit(),
    ],
    ids=repr,
)
def fan_init(request):
    "Inits that need tensors with at least 2 dims, so they can't run on `model`."
    return request.param


def test_original_untouched(model, init):
    old = _tensors(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}

    ModuleRebuilder(init)(model)

    assert all(a is b for a, b in zip(_tensors(model), old))
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_structure(model, init):
    new = ModuleRebuilder(init)(model)

    assert new.emb.weight is new.head.weight
    assert not new.frozen.requires_grad
    assert "tmp" in new._non_persistent_buffers_set
    assert new.state_dict().keys() == model.state_dict().keys()
    for a, b in zip(_tensors(model), _tensors(new)):
        assert a.data_ptr() != b.data_ptr()
        assert (a.shape, a.dtype, a.device) == (b.shape, b.dtype, b.device)


def test_fan_init(fan_init):
    new = ModuleRebuilder(fan_init)(nn.Linear(64, 32, bias=False))
    assert torch.isfinite(new.weight).all()


def test_fake_mode(model, init):
    with FakeTensorMode():
        fake = ModuleRebuilder(init)(model)

    assert all(is_fake(t) for t in _tensors(fake))
    assert fake.emb.weight is fake.head.weight


def test_fake_to_real(model):
    with FakeTensorMode():
        fake = ModuleRebuilder()(model)

    real = ModuleRebuilder(OnesInit())(fake)
    assert all(torch.all(t == 1) for t in _tensors(real))


def test_init_error_names_the_tensor():
    with pytest.raises(InitError, match="'bias'") as e:
        ModuleRebuilder(XavierUniformInit())(nn.Linear(4, 4))

    assert isinstance(e.value.__cause__, ValueError)


def test_allow_failure():
    new = ModuleRebuilder(XavierUniformInit(allow_failure=True))(nn.Linear(4, 4))

    # The 2-D weight is initialized, the 1-D bias fails and is skipped.
    assert torch.isfinite(new.weight).all()
