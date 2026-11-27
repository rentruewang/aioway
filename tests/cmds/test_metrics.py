# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

import pytest
import torch
from torch import testing as tt
from torch.nn import functional as F

from aioway.cmds import (
    Accuracy,
    ColStat,
    ColStatDomain,
    ColStatMetric,
    ConfusionMatrix,
    F1Score,
    FBetaScore,
    MaxMetric,
    MeanMetric,
    Metric,
    MinMetric,
    Precision,
    Recall,
    StdMetric,
    SumMetric,
    VarMetric,
)
from aioway.cmds.metrics._utils import as_buffer
from aioway.t import (
    Domain,
    InvalidValueForDomain,
    Pair,
    ScoreTensorDomain,
    TensorDomain,
)


class _Case(typing.NamedTuple):
    metric: cabc.Callable[[], Metric]
    "Makes a fresh metric."

    batches: list
    "Fed to `update` in order."

    expected: typing.Any
    "What `compute` should give."


def _run(case: _Case) -> None:
    metric = case.metric()

    for batch in case.batches:
        metric.update(batch)

    expected = torch.tensor(case.expected, dtype=torch.float)
    tt.assert_close(metric.compute().float(), expected, atol=1e-4, rtol=1e-5)


# Base ====


class _CountMetric(Metric[torch.Tensor, torch.Tensor]):
    "Ad hoc: how many elements were seen."

    def __init__(self) -> None:
        super().__init__()
        self.count = as_buffer(torch.tensor(0))

    @typing.override
    def _input_domain(self) -> Domain[torch.Tensor]:
        return TensorDomain()

    @typing.override
    def _output_domain(self) -> Domain[torch.Tensor]:
        return TensorDomain()

    @typing.override
    def _update(self, batch: torch.Tensor, /) -> None:
        self.count += batch.numel()

    @typing.override
    def _compute(self) -> torch.Tensor:
        return self.count


class _WrongOutputMetric(_CountMetric):
    "Ad hoc: claims a float score, but computes an int count."

    @typing.override
    def _output_domain(self) -> Domain[torch.Tensor]:
        return ScoreTensorDomain()


class TestMetric:
    @pytest.fixture
    def metric(self):
        return _CountMetric()

    def test_forward_is_running(self, metric: _CountMetric):
        assert metric(torch.zeros(3)) == 3
        assert metric(torch.zeros(2)) == 5

    def test_update_then_compute(self, metric: _CountMetric):
        metric.update(torch.zeros(3))
        metric.update(torch.zeros(4))
        assert metric.compute() == 7

    def test_update_checks_input(self, metric: _CountMetric):
        with pytest.raises(InvalidValueForDomain):
            metric.update("not a tensor")  # type: ignore

    def test_compute_checks_output(self):
        metric = _WrongOutputMetric()
        metric.update(torch.zeros(3))

        with pytest.raises(InvalidValueForDomain):
            metric.compute()

    def test_state_dict_round_trip(self, metric: _CountMetric):
        metric.update(torch.zeros(3))

        loaded = _CountMetric()
        loaded.load_state_dict(metric.state_dict())
        assert loaded.compute() == 3


# Arithmetics ====


def _arith_cases():
    batches = [torch.tensor([1.0, 2.0]), torch.tensor([[3.0], [6.0]])]
    yield _Case(SumMetric, batches, 12.0)
    yield _Case(MeanMetric, batches, 3.0)
    yield _Case(MaxMetric, batches, 6.0)
    yield _Case(MinMetric, batches, 1.0)
    yield _Case(VarMetric, batches, 14 / 3)
    yield _Case(StdMetric, batches, (14 / 3) ** 0.5)

    ints = [torch.tensor([1, 2, 3])]
    yield _Case(SumMetric, ints, 6.0)
    yield _Case(MeanMetric, ints, 2.0)
    yield _Case(VarMetric, ints, 1.0)
    yield _Case(StdMetric, ints, 1.0)


class TestAriths:
    @pytest.fixture(params=_arith_cases())
    def case(self, request) -> _Case:
        return request.param

    def test_metric(self, case: _Case):
        _run(case)


# Classification ====


def _clf_cases():
    # The example in `torchmetrics`' `MulticlassF1Score` docs.
    preds = torch.tensor([2, 1, 0, 1])
    target = torch.tensor([2, 1, 0, 0])
    pair = Pair(input=preds, target=target, batch_size=[4])
    logits = Pair(input=F.one_hot(preds, 3).float(), target=target, batch_size=[4])
    split = [pair[:2], pair[2:]]

    conf_mat = lambda: ConfusionMatrix(3)
    cm = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    yield _Case(conf_mat, [pair], cm)
    yield _Case(conf_mat, [logits], cm)
    yield _Case(conf_mat, split, cm)

    f1 = lambda: F1Score(3)
    yield _Case(f1, [pair], 0.7778)
    yield _Case(f1, split, 0.7778)
    yield _Case(lambda: F1Score(3, average="none"), [pair], [2 / 3, 2 / 3, 1])
    yield _Case(lambda: F1Score(3, average="micro"), [pair], 0.75)
    yield _Case(lambda: F1Score(3, average="weighted"), [pair], 0.75)
    yield _Case(lambda: FBetaScore(3, beta=1.0), [pair], 0.7778)

    yield _Case(lambda: Accuracy(3, average="micro"), [pair], 0.75)
    yield _Case(lambda: Accuracy(3), [pair], 0.8333)
    yield _Case(lambda: Precision(3), [pair], 0.8333)
    yield _Case(lambda: Recall(3), [pair], 0.8333)


class TestClfs:
    @pytest.fixture(params=_clf_cases())
    def case(self, request) -> _Case:
        return request.param

    def test_metric(self, case: _Case):
        _run(case)


# Stats ====


def _col_stat_cases():
    yield [torch.tensor([1.0, 2.0]), torch.tensor([[3.0], [6.0]])]
    yield [torch.tensor([1, 2, 3])]
    yield list(torch.randn(1000).split(100))


class TestColStat:
    @pytest.fixture(params=_col_stat_cases())
    def batches(self, request) -> list[torch.Tensor]:
        return request.param

    def test_metric(self, batches: list[torch.Tensor]):
        metric = ColStatMetric()

        for batch in batches:
            metric.update(batch)

        result = metric.compute()
        values = torch.cat([batch.flatten() for batch in batches]).double()

        assert result.count == values.numel()
        tt.assert_close(result.avg.double(), values.mean(), atol=1e-4, rtol=1e-5)
        tt.assert_close(result.std_dev.double(), values.std(), atol=1e-4, rtol=1e-5)
        tt.assert_close(result.low.double(), values.min(), atol=1e-4, rtol=1e-5)
        tt.assert_close(result.high.double(), values.max(), atol=1e-4, rtol=1e-5)

    def test_domain_accepts_col_stat(self):
        stat = _col_stat(count=torch.tensor(3))
        assert stat in ColStatDomain()

    def test_domain_rejects_float_count(self):
        stat = _col_stat(count=torch.tensor(3.0))
        assert stat not in ColStatDomain()

    def test_domain_rejects_non_col_stat(self):
        t: typing.Any = torch.tensor(1.0)
        assert t not in ColStatDomain()


def _col_stat(count: torch.Tensor) -> ColStat:
    one = torch.tensor(1.0)
    return ColStat(count=count, avg=one, std_dev=one, low=one, high=one, batch_size=[])
