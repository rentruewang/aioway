# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
import tensordict as td
import torch

from aioway.t import (
    AnyDomain,
    DTypeFamilyMixin,
    FloatTensorDomain,
    IntTensorDomain,
    InvalidValueForDomain,
    ListDomain,
    Pair,
    PairDomain,
    RangeMixin,
    RequiresGradMixin,
    ScoreTensorDomain,
    SizedListDomain,
    SizedSpatialDomain,
    SpatialDomain,
    TDictDomain,
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
            (sized_image(3, 4, 5), ListDomain()),
            (sized_image(3, 4, 5), TensorDomain()),
            (sized_video(2, 3, 4, 5), video()),
            (TemporalDomain(), ListDomain()),
            (SpatialDomain(), ListDomain()),
            (SizedSpatialDomain(length=4), SpatialDomain()),
            (SizedSpatialDomain(length=4), SizedListDomain(length=4)),
            (ListDomain(SpatialDomain()), ListDomain(ListDomain())),
            (grid_of(), ListDomain(ListDomain())),
            (video(), TemporalDomain()),
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
            (ListDomain(), TemporalDomain()),
            (SizedSpatialDomain(length=4), SizedSpatialDomain(length=5)),
            (ListDomain(ListDomain()), grid_of()),
            (TensorDomain(), ListDomain()),
        ],
    )
    def test_not_sub_domain(self, domain, other):
        assert not (domain <= other)


class TestShape:
    def test_sized_image(self):
        assert sized_image(3, 4, 5).shape == (3, 4, 5)
        assert sized_image(3, 4, 5).ndim == 3

    def test_sized_video(self):
        assert sized_video(2, 3, 4, 5).shape == (2, 3, 4, 5)

    def test_unsized_inner(self):
        assert SizedListDomain(ListDomain(), length=2).shape == (2, -1)

    @pytest.mark.parametrize("batch", [(), (2,), (6, 2)])
    def test_batch_size_of(self, batch):
        value = torch.rand(*batch, 3, 4, 5)
        assert sized_image(3, 4, 5).batch_size_of(value) == batch


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
            (sized_image(3, 4, 5), torch.rand(3, 4, 5)),
            (sized_image(3, 4, 5), torch.rand(6, 2, 3, 4, 5)),
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
            (image(), torch.rand(4, 5)),
            (sized_image(3, 4, 5), torch.rand(2, 3, 5, 4)),
            (video(), torch.rand(3, 4, 5)),
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


# Tensor mixins ====


class _FloatList(DTypeFamilyMixin, ListDomain):
    "A float list, to check the mixin composes with lists."


class TestTensorMixins:
    def test_dtype(self):
        DTypeFamilyMixin(family="float").check(torch.rand(2))
        DTypeFamilyMixin(family="int").check(torch.arange(2))

        with pytest.raises(InvalidValueForDomain):
            DTypeFamilyMixin(family="float").check(torch.arange(2))

    def test_dtype_fake(self):
        DTypeFamilyMixin(family="float").check(_fake(2))

    def test_dtype_with_list(self):
        domain = _FloatList(SizedListDomain(length=3), family="float")
        domain.check(torch.rand(2, 3))

        with pytest.raises(InvalidValueForDomain):
            domain.check(torch.arange(6).view(2, 3))

        with pytest.raises(InvalidValueForDomain):
            domain.check(torch.rand(2, 4))

    def test_requires_grad(self):
        RequiresGradMixin(requires_grad=True).check(torch.rand(2, requires_grad=True))
        RequiresGradMixin(requires_grad=False).check(torch.rand(2))

        with pytest.raises(InvalidValueForDomain):
            RequiresGradMixin(requires_grad=True).check(torch.rand(2))

    def test_range(self):
        RangeMixin(low=0, high=1).check(torch.rand(2))

        with pytest.raises(InvalidValueForDomain):
            RangeMixin(low=0, high=1).check(torch.full((2,), 2.0))

    def test_range_fake_skipped(self):
        RangeMixin(low=0, high=1).check(_fake(2))


class TestDtypes:
    def test_int_float(self):
        IntTensorDomain().check(torch.arange(2))
        FloatTensorDomain().check(torch.rand(2))

        with pytest.raises(InvalidValueForDomain):
            IntTensorDomain().check(torch.rand(2))

        with pytest.raises(InvalidValueForDomain):
            FloatTensorDomain().check(torch.arange(2))

    def test_score(self):
        ScoreTensorDomain().check(torch.tensor(1.0))

        with pytest.raises(InvalidValueForDomain):
            ScoreTensorDomain().check(torch.rand(1))

        with pytest.raises(InvalidValueForDomain):
            ScoreTensorDomain().check(torch.tensor(1))

    def test_hierarchy(self):
        assert ScoreTensorDomain() <= FloatTensorDomain()
        assert FloatTensorDomain() <= TensorDomain()
        assert not (FloatTensorDomain() <= IntTensorDomain())


# Pairs ====


def test_pair_domain():
    PairDomain().check(
        Pair(input=torch.rand(2, 3), target=torch.rand(2), batch_size=[2])
    )

    with pytest.raises(InvalidValueForDomain):
        tensor: typing.Any = torch.randn(2, 3)
        PairDomain().check(tensor)


def test_pair_domain_fields():
    pair = Pair(input=torch.rand(2, 3), target=torch.arange(2), batch_size=[2])
    PairDomain(FloatTensorDomain(), IntTensorDomain()).check(pair)

    with pytest.raises(InvalidValueForDomain):
        PairDomain(IntTensorDomain(), IntTensorDomain()).check(pair)

    assert PairDomain(FloatTensorDomain(), IntTensorDomain()) <= PairDomain()
    assert not (PairDomain() <= PairDomain(FloatTensorDomain(), IntTensorDomain()))


# Dicts ====


class TestDictDomain:
    def test_check(self):
        tdict = td.TensorDict(x=torch.rand(2, 3), y=torch.arange(2), batch_size=[2])
        TDictDomain({"x": FloatTensorDomain()}).check(tdict)

        with pytest.raises(InvalidValueForDomain):
            TDictDomain({"x": IntTensorDomain()}).check(tdict)

        with pytest.raises(InvalidValueForDomain):
            TDictDomain({"z": TensorDomain()}).check(tdict)

    def test_hierarchy(self):
        floats = TDictDomain({"x": FloatTensorDomain()})
        tensors = TDictDomain({"x": TensorDomain()})

        assert floats <= tensors
        assert not (tensors <= floats)
        assert not (floats <= TDictDomain({"y": TensorDomain()}))


# Hashing ====


@pytest.fixture(
    params=[
        lambda: AnyDomain(),
        lambda: TensorDomain(),
        lambda: FloatTensorDomain(),
        lambda: ScoreTensorDomain(),
        lambda: sized_image(3, 4, 5),
        lambda: video(),
        lambda: PairDomain(FloatTensorDomain(), IntTensorDomain()),
        lambda: TDictDomain({"x": FloatTensorDomain(), "y": sized_image(3, 4, 5)}),
    ]
)
def make_domain(request) -> typing.Callable[[], typing.Any]:
    "Builds a new, equal domain on every call."

    return request.param


def test_equal_domains_hash_equal(make_domain):
    left, right = make_domain(), make_domain()

    assert left is not right
    assert left == right
    assert hash(left) == hash(right)
    assert len({left, right}) == 1
