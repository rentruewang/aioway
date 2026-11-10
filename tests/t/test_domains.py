# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
import torch
from torch._subclasses import fake_tensor
from torchvision import tv_tensors as tvt

from aioway.t import (
    GridDomain,
    ImageDomain,
    InvalidDomainValue,
    Pair,
    PairDomain,
    SizedGridDomain,
    SizedImageDomain,
    SizedVideoDomain,
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


def test_sized_image_is_under_its_ancestors():
    image = SizedImageDomain(4, 5)
    for ancestor in [
        image,
        ImageDomain(),
        SizedGridDomain(4, 5),
        GridDomain(),
        TensorDomain(),
    ]:
        assert image.sub_domain_of(ancestor)


def test_sub_domain_is_not_symmetric():
    assert not ImageDomain().sub_domain_of(SizedImageDomain(4, 5))
    assert not GridDomain().sub_domain_of(ImageDomain())


def test_unrelated_domains():
    assert not SizedImageDomain(4, 5).sub_domain_of(SizedGridDomain(5, 4))
    assert not SizedImageDomain(4, 5).sub_domain_of(VideoDomain())


# Real values ====


def test_accepts_real_values():
    TensorDomain().check(torch.rand(2))
    GridDomain().check(torch.rand(3, 4, 5))
    SizedImageDomain(4, 5).check(tvt.Image(torch.rand(3, 4, 5)))
    SizedVideoDomain(2).check(tvt.Video(torch.rand(2, 3, 4, 5)))


@pytest.mark.parametrize(
    "domain, value",
    [
        (TensorDomain(), [1, 2]),
        (GridDomain(), torch.rand(4, 5)),
        (SizedGridDomain(4, 5), torch.rand(3, 5, 4)),
        (ImageDomain(), torch.rand(3, 4, 5)),
        (ImageDomain(), tvt.Image(torch.full((3, 4, 5), 2.0))),
        (VideoDomain(), tvt.Image(torch.rand(3, 4, 5))),
        (SizedVideoDomain(3), tvt.Video(torch.rand(2, 3, 4, 5))),
    ],
)
def test_rejects_real_values(domain, value):
    with pytest.raises(InvalidDomainValue):
        domain.check(value)


# Fake values ====


def test_fakes_skip_data_checks():
    ImageDomain().check(fake(3, 4, 5))
    SizedImageDomain(4, 5).check(fake(3, 4, 5))
    VideoDomain().check(fake(2, 3, 4, 5))


def test_fakes_still_check_shapes():
    with pytest.raises(InvalidDomainValue):
        GridDomain().check(fake(4, 5))

    with pytest.raises(InvalidDomainValue):
        SizedImageDomain(4, 5).check(fake(3, 5, 4))


# Pairs ====


def test_pair_domain():
    PairDomain().check(
        Pair(input=torch.rand(2, 3), target=torch.rand(2), batch_size=[2])
    )

    with pytest.raises(InvalidDomainValue):
        tensor: typing.Any = torch.randn(2, 3)
        PairDomain().check(tensor)
