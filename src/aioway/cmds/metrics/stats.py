# Copyright (c) AIoWay Authors - All Rights Reserved

"Describe the data seen, like `pandas.DataFrame.describe`."

import typing
from collections import abc as cabc

import tensordict as td
import torch
from torch import nn

from aioway.t import Domain, TDictDomain, TensorDomain

from .ariths import CountMetric, MaxMetric, MeanMetric, MinMetric, StdMetric
from .metrics import Metric

__all__ = ["ColStatMetric"]


class ColStatMetric(Metric[torch.Tensor, td.TensorDict]):
    """
    Many metrics on the same tensor, the results in a `TensorDict` by name.

    Defaults to a column of `pd.DataFrame.describe`, without percentiles.
    """

    def __init__(self, metrics: dict[str, Metric] | None = None) -> None:
        super().__init__()

        if metrics is None:
            metrics = {
                "count": CountMetric(),
                "avg": MeanMetric(),
                "std_dev": StdMetric(),
                "low": MinMetric(),
                "high": MaxMetric(),
            }

        for key, val in metrics.items():
            if not isinstance(val, Metric):
                raise ValueError(
                    f"Must be a dict of metrics. But got {val} at {key!r}."
                )

        module: typing.Any = nn.ModuleDict(metrics)
        self.metrics: cabc.Mapping[str, Metric] = module

    @typing.override
    def _input_domain(self) -> Domain[torch.Tensor]:
        return TensorDomain()

    @typing.override
    def _output_domain(self) -> Domain[td.TensorDict]:
        return TDictDomain({name: m.__domain__ for name, m in self.metrics.items()})

    @typing.override
    def _update(self, batch: torch.Tensor, /) -> None:
        for metric in self.metrics.values():
            metric.update(batch)

    @typing.override
    def _compute(self) -> td.TensorDict:
        return td.TensorDict(
            {name: m.compute() for name, m in self.metrics.items()}, batch_size=[]
        )
