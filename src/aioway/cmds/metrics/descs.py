# Copyright (c) AIoWay Authors - All Rights Reserved

"Describe the data seen, like `pandas.DataFrame.describe`, but the result is a `Domain`."

import typing

import tensordict as td
import torch
from torch import nn

from aioway.t import (
    Domain,
    JointDomain,
    RangeMixin,
    ScalarDomain,
    TDictDomain,
    TensorDomain,
    tcol_to_tdict,
)

from ._utils import as_buffer
from .ariths import MaxMetric, MinMetric
from .metrics import Metric

__all__ = ["DescribeDomain", "DescribeMetric", "MaybeNestedDescribeMetric"]


class DescribeDomain(RangeMixin):
    "Tensors whose values are in `[low, high]`, with the stats of the data it describes."

    count: int
    "The number of values seen."

    mean: float
    "The mean of the values."

    std: float
    "The sample standard deviation of the values, same as `pandas`."

    @property
    def __extends__(self) -> list[Domain]:
        return [TensorDomain()]


class DescribeMetric(Metric[torch.Tensor, DescribeDomain]):
    "Describe all values of all batches of a tensor."

    def __init__(self) -> None:
        super().__init__()
        self.count = as_buffer(torch.tensor(0))
        self.total = as_buffer(torch.tensor(0.0, dtype=torch.double))
        self.sq_total = as_buffer(torch.tensor(0.0, dtype=torch.double))
        self.low = MinMetric()
        self.high = MaxMetric()

    @typing.override
    def _input_domain(self) -> Domain[torch.Tensor]:
        return TensorDomain()

    @typing.override
    def _output_domain(self) -> Domain[DescribeDomain]:
        return ScalarDomain(DescribeDomain)

    @typing.override
    def _update(self, batch: torch.Tensor, /) -> None:
        values = batch.detach().double()
        self.count += values.numel()
        self.total += values.sum()
        self.sq_total += values.square().sum()
        self.low._update(batch)
        self.high._update(batch)

    @typing.override
    def _compute(self) -> DescribeDomain:
        count: typing.Any = self.count.item()
        mean = self.total / count
        var = (self.sq_total - count * mean**2) / (count - 1)
        return DescribeDomain(
            count=count,
            mean=mean.item(),
            std=var.clamp(min=0).sqrt().item(),
            low=self.low.compute().item(),
            high=self.high.compute().item(),
        )


class MaybeNestedDescribeMetric(Metric[typing.Any, Domain]):
    """
    `DescribeMetric` for a tensor, a `TDictDomain` of describes for a tensor collection.

    Collections nest, keys are found on the first batch.
    """

    def __init__(self) -> None:
        super().__init__()
        self.leaf: DescribeMetric | None = None
        self.children_: nn.ModuleDict | None = None

    @typing.override
    def _input_domain(self) -> Domain:
        return JointDomain((TensorDomain(), TDictDomain()))

    @typing.override
    def _output_domain(self) -> Domain[Domain]:
        return ScalarDomain(Domain)

    @typing.override
    def _update(self, batch: typing.Any, /) -> None:
        if not td.is_tensor_collection(batch):
            if self.leaf is None:
                self.leaf = DescribeMetric()
            self.leaf.update(batch)
            return

        tdict = tcol_to_tdict(batch)
        if self.children_ is None:
            self.children_ = nn.ModuleDict(
                {key: MaybeNestedDescribeMetric() for key in tdict.keys()}
            )

        for key, child in self.children_.items():
            child.update(tdict[key])

    @typing.override
    def _compute(self) -> Domain:
        if self.leaf is not None:
            return self.leaf.compute()

        assert self.children_ is not None, "Nothing seen yet."
        return TDictDomain(
            {key: child.compute() for key, child in self.children_.items()}
        )
