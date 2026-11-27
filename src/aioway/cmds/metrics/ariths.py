# Copyright (c) AIoWay Authors - All Rights Reserved

"Metrics that reduce all values of all batches into one float."

import abc
import typing

import torch

from aioway.t import Domain, IntTensorDomain, ScoreTensorDomain, TensorDomain

from ._utils import as_buffer
from .metrics import Metric

__all__ = [
    "CountMetric",
    "SumMetric",
    "MeanMetric",
    "VarMetric",
    "StdMetric",
    "MaxMetric",
    "MinMetric",
]


class _ArithMetric(Metric[torch.Tensor, torch.Tensor], abc.ABC):
    "Any tensor in, a float out."

    @typing.override
    def _input_domain(self) -> Domain[torch.Tensor]:
        return TensorDomain()

    @typing.override
    def _output_domain(self) -> Domain[torch.Tensor]:
        return ScoreTensorDomain()


class CountMetric(_ArithMetric):
    "The number of values."

    def __init__(self) -> None:
        super().__init__()
        self.count = as_buffer(torch.tensor(0))

    @typing.override
    def _output_domain(self) -> Domain[torch.Tensor]:
        return IntTensorDomain()

    @typing.override
    def _update(self, batch: torch.Tensor, /) -> None:
        self.count += batch.numel()

    @typing.override
    def _compute(self) -> torch.Tensor:
        return self.count


class SumMetric(_ArithMetric):
    "The sum of all values."

    def __init__(self) -> None:
        super().__init__()
        self.total = as_buffer(torch.tensor(0.0))

    @typing.override
    def _update(self, batch: torch.Tensor, /) -> None:
        self.total += batch.detach().float().sum()

    @typing.override
    def _compute(self) -> torch.Tensor:
        return self.total


class MeanMetric(_ArithMetric):
    "The mean of all values."

    def __init__(self) -> None:
        super().__init__()
        self.total = as_buffer(torch.tensor(0.0))
        self.count = as_buffer(torch.tensor(0))

    @typing.override
    def _update(self, batch: torch.Tensor, /) -> None:
        self.total += batch.detach().float().sum()
        self.count += batch.numel()

    @typing.override
    def _compute(self) -> torch.Tensor:
        return self.total / self.count


class VarMetric(_ArithMetric):
    "The sample variance of all values, same as `pandas`."

    def __init__(self) -> None:
        super().__init__()
        self.count = as_buffer(torch.tensor(0))
        self.total = as_buffer(torch.tensor(0.0, dtype=torch.double))
        self.sq_total = as_buffer(torch.tensor(0.0, dtype=torch.double))

    @typing.override
    def _update(self, batch: torch.Tensor, /) -> None:
        values = batch.detach().double()
        self.count += values.numel()
        self.total += values.sum()
        self.sq_total += values.square().sum()

    @typing.override
    def _compute(self) -> torch.Tensor:
        mean = self.total / self.count
        var = (self.sq_total - self.count * mean**2) / (self.count - 1)
        return var.clamp(min=0)


class StdMetric(VarMetric):
    "The sample standard deviation of all values, same as `pandas`."

    @typing.override
    def _compute(self) -> torch.Tensor:
        return super()._compute().sqrt()


class MaxMetric(_ArithMetric):
    "The largest value."

    def __init__(self) -> None:
        super().__init__()
        self.value = as_buffer(torch.tensor(-torch.inf))

    @typing.override
    def _update(self, batch: torch.Tensor, /) -> None:
        self.value = torch.maximum(self.value, batch.detach().float().max())

    @typing.override
    def _compute(self) -> torch.Tensor:
        return self.value


class MinMetric(_ArithMetric):
    "The smallest value."

    def __init__(self) -> None:
        super().__init__()
        self.value = as_buffer(torch.tensor(torch.inf))

    @typing.override
    def _update(self, batch: torch.Tensor, /) -> None:
        self.value = torch.minimum(self.value, batch.detach().float().min())

    @typing.override
    def _compute(self) -> torch.Tensor:
        return self.value
