# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing

from aioway.t import Domain

__all__ = ["Metric"]


class Metric[I, O](abc.ABC):
    """
    The base class for metrics.

    The API mimics `torchmetrics`, because

    1. It seems somewhat abandoned.
    2. Lots of metrics are ~10 lines, where wiring takes more code.
    """

    def __call__(self, batch: I, /) -> O:
        self.update(batch)
        return self.compute()

    @property
    def __domain__(self) -> Domain[O]:
        return self._output_domain()

    def compute(self) -> O:
        result = self._compute()
        self.__domain__.check(result)
        return result

    def update(self, batch: I, /) -> None:
        self._input_domain().check(batch)
        self._update(batch)

    @abc.abstractmethod
    def _output_domain(self) -> Domain[O]:
        raise NotImplementedError

    @abc.abstractmethod
    def _input_domain(self) -> Domain[I]:
        raise NotImplementedError

    @abc.abstractmethod
    def _update(self, batch: I, /) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _compute(self) -> O:
        raise NotImplementedError
