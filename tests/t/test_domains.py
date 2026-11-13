# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
import torch

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
    fake_mode,
)


def _fake(*shape: int) -> torch.Tensor:
    with fake_mode():
        return torch.empty(*shape)


class TestHierarchy:
    def test_equal_domains_same_type(self):
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
    def test_sub_domain(self, domain, ancestor):
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
    def test_not_sub_domain(self, domain, other):
        assert not domain.sub_domain_of(other)


class TestRealValues:
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
    def test_accepts(self, domain, value):
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
    def test_rejects(self, domain, value):
        with pytest.raises(InvalidDomain):
            domain.check(value)
        assert value not in domain


class TestFakeValues:
    def test_skip_data_checks(self):
        ImageDomain().check(_fake(2, 4, 5, 3))
        SizedImageDomain(4, 5).check(_fake(2, 4, 5, 3))
        VideoDomain().check(_fake(1, 2, 4, 5, 3))

    def test_still_check_shapes(self):
        with pytest.raises(InvalidDomain):
            GridDomain().check(_fake(4, 5))

        with pytest.raises(InvalidDomain):
            SizedImageDomain(4, 5).check(_fake(2, 5, 4, 3))


# Pairs ====


def test_pair_domain():
    PairDomain().check(
        Pair(input=torch.rand(2, 3), target=torch.rand(2), batch_size=[2])
    )

    with pytest.raises(InvalidDomain):
        tensor: typing.Any = torch.randn(2, 3)
        PairDomain().check(tensor)
