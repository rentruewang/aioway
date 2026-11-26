# Copyright (c) AIoWay Authors - All Rights Reserved

"Metrics that reduce all values of all batches into one float."

import abc
import typing

import torch
from torch import nn

from aioway.t import Domain, ScoreTensorDomain, TensorDomain

from .metrics import Metric

__all__ = ["SumMetric", "MeanMetric", "MaxMetric", "MinMetric"]


class _ArithMetric(Metric[torch.Tensor, torch.Tensor], abc.ABC):
    "Any tensor in, a float out."

    @typing.override
    def _input_domain(self) -> Domain[torch.Tensor]:
        return TensorDomain()

    @typing.override
    def _output_domain(self) -> Domain[torch.Tensor]:
        return ScoreTensorDomain()


class SumMetric(_ArithMetric):
    "The sum of all values."

    def __init__(self) -> None:
        super().__init__()
        self.total: torch.Tensor = nn.Buffer(torch.tensor(0.0))

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
        self.total: torch.Tensor = nn.Buffer(torch.tensor(0.0))
        self.count: torch.Tensor = nn.Buffer(torch.tensor(0))

    @typing.override
    def _update(self, batch: torch.Tensor, /) -> None:
        self.total += batch.detach().float().sum()
        self.count += batch.numel()

    @typing.override
    def _compute(self) -> torch.Tensor:
        return self.total / self.count


class MaxMetric(_ArithMetric):
    "The largest value."

    def __init__(self) -> None:
        super().__init__()
        self.value: torch.Tensor = nn.Buffer(torch.tensor(-torch.inf))

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
        self.value: torch.Tensor = nn.Buffer(torch.tensor(torch.inf))

    @typing.override
    def _update(self, batch: torch.Tensor, /) -> None:
        self.value = torch.minimum(self.value, batch.detach().float().min())

    @typing.override
    def _compute(self) -> torch.Tensor:
        return self.value
