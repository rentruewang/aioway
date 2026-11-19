# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
import torch

from aioway.t import (
    AnyDomain,
    BatchDomain,
    InvalidValueForDomain,
    ListDomain,
    Pair,
    PairDomain,
    SizedBatchDomain,
    SizedListDomain,
    SizedSpatialDomain,
    SpatialDomain,
    TemporalDomain,
    TensorDomain,
    fake_mode,
    grid_of,
    image,
    sized_image,
    sized_video,
    video,
)


def _fake(*shape: int) -> torch.Tensor:
    with fake_mode():
        return torch.empty(*shape)


class TestHierarchy:
    def test_equal_domains_same_type(self):
        assert SizedListDomain(length=4) == SizedListDomain(length=4)
        assert len({SizedListDomain(length=4), SizedListDomain(length=4)}) == 1
        assert SizedListDomain(length=4) != SizedListDomain(length=5)

    @pytest.mark.parametrize(
        "domain, ancestor",
        [
            (sized_image(3, 4, 5), sized_image(3, 4, 5)),
            (sized_image(3, 4, 5), image()),
            (sized_image(3, 4, 5), BatchDomain()),
            (sized_image(3, 4, 5), TensorDomain()),
            (sized_video(2, 3, 4, 5), video()),
            (BatchDomain(), ListDomain()),
            (TemporalDomain(), ListDomain()),
            (SpatialDomain(), ListDomain()),
            (SizedSpatialDomain(length=4), SpatialDomain()),
            (SizedBatchDomain(length=2), BatchDomain()),
            (SizedSpatialDomain(length=4), SizedListDomain(length=4)),
            (ListDomain(SpatialDomain()), ListDomain(ListDomain())),
            (grid_of(), ListDomain(ListDomain())),
            (video(), BatchDomain(TemporalDomain())),
            (video(), AnyDomain()),
            (ListDomain(PairDomain()), ListDomain()),
        ],
    )
    def test_sub_domain(self, domain, ancestor):
        assert domain <= ancestor

    @pytest.mark.parametrize(
        "domain, other",
        [
            (image(), sized_image(3, 4, 5)),
            (sized_image(3, 4, 5), sized_image(3, 5, 4)),
            (sized_image(3, 4, 5), video()),
            (TemporalDomain(), SpatialDomain()),
            (ListDomain(), BatchDomain()),
            (SizedSpatialDomain(length=4), SizedSpatialDomain(length=5)),
            (ListDomain(ListDomain()), grid_of()),
            (TensorDomain(), ListDomain()),
        ],
    )
    def test_not_sub_domain(self, domain, other):
        assert not (domain <= other)


class TestShape:
    def test_sized_image(self):
        assert sized_image(3, 4, 5).shape == (-1, 3, 4, 5)
        assert sized_image(3, 4, 5).ndim == 4

    def test_sized_video(self):
        assert sized_video(2, 3, 4, 5).shape == (-1, 2, 3, 4, 5)

    def test_sized_batch(self):
        assert SizedBatchDomain(sized_image(3, 4, 5).inner, length=2).shape == (
            2,
            3,
            4,
            5,
        )


class TestRealValues:
    @pytest.mark.parametrize(
        "domain, value",
        [
            (AnyDomain(), "anything"),
            (TensorDomain(), torch.rand(2)),
            (ListDomain(), torch.rand(2)),
            (ListDomain(ListDomain()), torch.rand(2, 3)),
            (grid_of(), torch.rand(4, 5)),
            (image(), torch.rand(2, 3, 4, 5)),
            (sized_image(3, 4, 5), torch.rand(2, 3, 4, 5)),
            (sized_video(2, 3, 4, 5), torch.rand(1, 2, 3, 4, 5)),
        ],
    )
    def test_accepts(self, domain, value):
        domain.check(value)
        assert value in domain

    @pytest.mark.parametrize(
        "domain, value",
        [
            (TensorDomain(), [1, 2]),
            (ListDomain(), torch.tensor(1.0)),
            (grid_of(), torch.rand(4)),
            (image(), torch.rand(2, 4, 5)),
            (sized_image(3, 4, 5), torch.rand(2, 3, 5, 4)),
            (video(), torch.rand(2, 3, 4, 5)),
            (sized_video(3, 3, 4, 5), torch.rand(1, 2, 3, 4, 5)),
        ],
    )
    def test_rejects(self, domain, value):
        with pytest.raises(InvalidValueForDomain):
            domain.check(value)
        assert value not in domain


class TestFakeValues:
    def test_accepts(self):
        image().check(_fake(2, 3, 4, 5))
        sized_image(3, 4, 5).check(_fake(2, 3, 4, 5))
        video().check(_fake(1, 2, 3, 4, 5))

    def test_still_check_shapes(self):
        with pytest.raises(InvalidValueForDomain):
            grid_of().check(_fake(4))

        with pytest.raises(InvalidValueForDomain):
            sized_image(3, 4, 5).check(_fake(2, 3, 5, 4))


# Pairs ====


def test_pair_domain():
    PairDomain().check(
        Pair(input=torch.rand(2, 3), target=torch.rand(2), batch_size=[2])
    )

    with pytest.raises(InvalidValueForDomain):
        tensor: typing.Any = torch.randn(2, 3)
        PairDomain().check(tensor)
