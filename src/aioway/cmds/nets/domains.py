# Copyright (c) AIoWay Authors - All Rights Reserved

"The domains emitters take: sized, with a dtype family, and maybe a range."

import dataclasses as dcls

from aioway.t import (
    DTypeFamily,
    DTypeFamilyMixin,
    RangeMixin,
    SizedListDomain,
    TensorItemDomain,
    sized_of,
)

__all__ = [
    "FloatListDomain",
    "FloatRangeListDomain",
    "IntRangeItemDomain",
    "float_list",
    "float_range_list",
    "int_range_item",
]


class FloatListDomain(DTypeFamilyMixin, SizedListDomain):
    "Float examples of a known shape."

    family: DTypeFamily = dcls.field(default="float", init=False)
    "Always float."


class FloatRangeListDomain(RangeMixin, FloatListDomain):
    "Float examples of a known shape, values in `[low, high]`."


class IntRangeItemDomain(RangeMixin, DTypeFamilyMixin, TensorItemDomain):
    "One int per example, values in `[low, high]`, e.g. a class id."

    family: DTypeFamily = dcls.field(default="int", init=False)
    "Always int."


def float_list(length: int, *shape: int) -> FloatListDomain:
    "Float examples of shape `[length, *shape]`."

    return FloatListDomain(sized_of(*shape), length=length)


def float_range_list(
    low: float, high: float, length: int, *shape: int
) -> FloatRangeListDomain:
    "Float examples of shape `[length, *shape]`, values in `[low, high]`."

    return FloatRangeListDomain(sized_of(*shape), length=length, low=low, high=high)


def int_range_item(low: int, high: int) -> IntRangeItemDomain:
    "One int per example, values in `[low, high]`."

    return IntRangeItemDomain(low=low, high=high)
