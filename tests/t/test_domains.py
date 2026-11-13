# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
import torch
from torch._subclasses import fake_tensor

from aioway.t import (
    AnyDomain,
    GridDomain,
    ImageDomain,
    InvalidDomain,
    ListDomain,
    Pair,
    PairDomain,
    SizedGridDomain,
    SizedImageDomain,
    SizedVideoDomain,
    SpatialDomain,
    TemporalDomain,
    TensorDomain,
    VideoDomain,
)


def fake(*shape: int) -> torch.Tensor:
    with fake_tensor.FakeTensorMode():
        return torch.empty(*shape)


# Hierarchy ====


def test_equal_domains_are_one_type():
    assert SizedGridDomain(4, 5) == SizedGridDomain(4, 5)
    assert len({SizedGridDomain(4, 5), SizedGridDomain(4, 5)}) == 1
    assert SizedGridDomain(4, 5) != SizedGridDomain(5, 4)


@pytest.mark.parametrize(
    "domain, ancestor",
    [
        (SizedImageDomain(4, 5), SizedImageDomain(4, 5)),
        (SizedImageDomain(4, 5), ImageDomain()),
        (SizedImageDomain(4, 5), SizedGridDomain(4, 5)),
        (SizedImageDomain(4, 5), GridDomain()),
        (SizedImageDomain(4, 5), TensorDomain()),
        (SizedVideoDomain(2), VideoDomain()),
        (TemporalDomain(), ListDomain()),
        (SpatialDomain(), ListDomain()),
        (ListDomain(ImageDomain()), ListDomain(GridDomain())),
        (GridDomain(), SpatialDomain(SpatialDomain())),
        (VideoDomain(), TemporalDomain(ImageDomain())),
        (VideoDomain(), ListDomain(GridDomain())),
        (VideoDomain(), AnyDomain()),
    ],
)
def test_sub_domain(domain, ancestor):
    assert domain.sub_domain_of(ancestor)


@pytest.mark.parametrize(
    "domain, other",
    [
        (ImageDomain(), SizedImageDomain(4, 5)),
        (GridDomain(), ImageDomain()),
        (SizedImageDomain(4, 5), SizedGridDomain(5, 4)),
        (SizedImageDomain(4, 5), VideoDomain()),
        (TemporalDomain(), SpatialDomain()),
        (ListDomain(), TemporalDomain()),
        (ListDomain(GridDomain()), ListDomain(ImageDomain())),
    ],
)
def test_not_sub_domain(domain, other):
    assert not domain.sub_domain_of(other)


# Real values ====


@pytest.mark.parametrize(
    "domain, value",
    [
        (AnyDomain(), "anything"),
        (TensorDomain(), torch.rand(2)),
        (ListDomain(), torch.rand(2, 3)),
        (ListDomain(ListDomain()), torch.rand(2, 3, 4)),
        (GridDomain(), torch.rand(2, 4, 5)),
        (SizedImageDomain(4, 5), torch.rand(2, 4, 5, 3)),
        (ImageDomain(), torch.full((2, 4, 5, 3), 255, dtype=torch.uint8)),
        (SizedVideoDomain(2), torch.rand(1, 2, 4, 5, 3)),
    ],
)
def test_accepts_real_values(domain, value):
    domain.check(value)
    assert value in domain


@pytest.mark.parametrize(
    "domain, value",
    [
        (TensorDomain(), [1, 2]),
        (ListDomain(), torch.rand(2)),
        (GridDomain(), torch.rand(4, 5)),
        (SizedGridDomain(4, 5), torch.rand(2, 5, 4, 3)),
        (ImageDomain(), torch.full((2, 4, 5, 3), 2.0)),
        (VideoDomain(), torch.rand(2, 3, 4)),
        (VideoDomain(), torch.full((1, 2, 4, 5, 3), 2.0)),
        (SizedVideoDomain(3), torch.rand(1, 2, 4, 5, 3)),
    ],
)
def test_rejects_real_values(domain, value):
    with pytest.raises(InvalidDomain):
        domain.check(value)
    assert value not in domain


# Fake values ====


def test_fakes_skip_data_checks():
    ImageDomain().check(fake(2, 4, 5, 3))
    SizedImageDomain(4, 5).check(fake(2, 4, 5, 3))
    VideoDomain().check(fake(1, 2, 4, 5, 3))


def test_fakes_still_check_shapes():
    with pytest.raises(InvalidDomain):
        GridDomain().check(fake(4, 5))

    with pytest.raises(InvalidDomain):
        SizedImageDomain(4, 5).check(fake(2, 5, 4, 3))


# Pairs ====


def test_pair_domain():
    PairDomain().check(
        Pair(input=torch.rand(2, 3), target=torch.rand(2), batch_size=[2])
    )

    with pytest.raises(InvalidDomain):
        tensor: typing.Any = torch.randn(2, 3)
        PairDomain().check(tensor)
