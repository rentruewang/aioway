# Copyright (c) AIoWay Authors - All Rights Reserved

"Describe the data seen, like `pandas.DataFrame.describe`, but the result is a `Domain`."

import typing

import tensordict as td
import torch

from aioway.t import (
    AnyDomain,
    Domain,
    FloatTensorDomain,
    IntTensorDomain,
    InvalidValueForDomain,
    TensorDomain,
)

from .ariths import MaxMetric, MeanMetric, MinMetric, StdMetric
from .metrics import Metric

__all__ = ["ColStat", "ColStatDomain", "ColStatMetric"]


class ColStat(td.TensorClass):
    "The stats of all values seen, like a column of `pandas.DataFrame.describe`."

    count: torch.Tensor
    "The number of values seen."

    avg: torch.Tensor
    "The mean of the values."

    std_dev: torch.Tensor
    "The sample standard deviation of the values, same as `pandas`."

    low: torch.Tensor
    "The smallest value."

    high: torch.Tensor
    "The largest value."


class ColStatDomain(Domain[ColStat]):
    "A `ColStat`: an int `count`, and a float item for each other stat."

    @property
    def __extends__(self) -> list[Domain]:
        return [AnyDomain()]

    def check_type(self, obj: ColStat, /) -> None:
        if not isinstance(obj, ColStat):
            raise InvalidValueForDomain(f"Expected a ColStat, got {type(obj)=}.")

        IntTensorDomain().check_type(obj.count)

        for stat in [obj.avg, obj.std_dev, obj.low, obj.high]:
            FloatTensorDomain().check_type(stat)


class ColStatMetric(Metric[torch.Tensor, ColStat]):
    "The `ColStat` of all values of all batches of a tensor."

    def __init__(self) -> None:
        super().__init__()
        self.avg = MeanMetric()
        self.std_dev = StdMetric()
        self.low = MinMetric()
        self.high = MaxMetric()

    @typing.override
    def _input_domain(self) -> Domain[torch.Tensor]:
        return TensorDomain()

    @typing.override
    def _output_domain(self) -> Domain[ColStat]:
        return ColStatDomain()

    @typing.override
    def _update(self, batch: torch.Tensor, /) -> None:
        self.avg.update(batch)
        self.std_dev.update(batch)
        self.low.update(batch)
        self.high.update(batch)

    @typing.override
    def _compute(self) -> ColStat:
        return ColStat(
            count=self.avg.count,
            avg=self.avg.compute(),
            std_dev=self.std_dev.compute(),
            low=self.low.compute(),
            high=self.high.compute(),
            batch_size=[],
        )
