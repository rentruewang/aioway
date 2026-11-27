# Copyright (c) AIoWay Authors - All Rights Reserved

import abc

from torch import nn

from aioway.t import Domain

__all__ = ["Metric"]


class Metric[I, O](nn.Module, abc.ABC):
    """
    The base class for metrics.

    The API mimics `torchmetrics`, because

    1. It seems somewhat abandoned.
    2. Lots of metrics are ~10 lines, where wiring takes more code.

    State is kept in `nn.Buffer`s, so `.to(device)` moves it.
    For a fresh state, create a new metric.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def __domain__(self) -> Domain[O]:
        "The domain of the computed result."

        return self._output_domain()

    def forward(self, batch: I, /) -> O:
        "Update with `batch`, then compute over everything seen so far."

        self.update(batch)
        return self.compute()

    def compute(self) -> O:
        "The result over all batches seen, checked against `__domain__`."

        result = self._compute()
        self.__domain__.check(result)
        return result

    def update(self, batch: I, /) -> None:
        "Check `batch` against the input domain, then add it to the state."

        self._input_domain().check(batch)
        self._update(batch)

    @abc.abstractmethod
    def _output_domain(self) -> Domain[O]:
        "The domain of what `compute` returns."

        raise NotImplementedError

    @abc.abstractmethod
    def _input_domain(self) -> Domain[I]:
        "The domain of the batches `update` takes."

        raise NotImplementedError

    @abc.abstractmethod
    def _update(self, batch: I, /) -> None:
        "Implementation of `update`."

        raise NotImplementedError

    @abc.abstractmethod
    def _compute(self) -> O:
        "Implementation of `compute`."

        raise NotImplementedError
