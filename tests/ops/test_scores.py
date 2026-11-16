# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import math
import typing

import pytest
import torch
from aioway.t import FloatTensorDomain, IntTensorDomain, TensorDomain
from torch import nn
from torch import testing as tt

from aioway.ops import (
    AccuracyOp,
    BinaryLossOp,
    ClassLossOp,
    Op,
    OpDomainMismatch,
    PairLossOp,
    RegressionLossOp,
    ScoreOp,
    TensorOp,
    TripletLossOp,
    ValueOp,
)


class _TypedTensorOp(Op[torch.Tensor]):
    "A tensor with a given domain, standing in for an `Op` whose domain is specific."

    tensor: torch.Tensor
    domain: TensorDomain

    @property
    @typing.override
    def __domain__(self) -> TensorDomain:
        return self.domain

    @typing.override
    def execute(self) -> torch.Tensor:
        return self.tensor

    @property
    @typing.override
    def __deps__(self) -> tuple[()]:
        return ()


def _floats(x: torch.Tensor) -> _TypedTensorOp:
    return _TypedTensorOp(x, FloatTensorDomain())


def _ints(x: torch.Tensor) -> _TypedTensorOp:
    return _TypedTensorOp(x, IntTensorDomain())


def _randn(*shape: int) -> _TypedTensorOp:
    return _floats(torch.randn(*shape))


def _bits(*shape: int) -> _TypedTensorOp:
    return _floats(torch.randint(0, 2, shape).float())


def _signs(*shape: int) -> _TypedTensorOp:
    return _floats(torch.randint(0, 2, shape).float() * 2 - 1)


def _classes(size: int, num_classes: int) -> _TypedTensorOp:
    return _ints(torch.randint(0, num_classes, (size,)))


def _scores():
    yield RegressionLossOp(nn.MSELoss(), _randn(8, 4), _randn(8, 4))
    yield BinaryLossOp(nn.BCEWithLogitsLoss(), _randn(8, 4), _bits(8, 4))
    yield ClassLossOp(nn.CrossEntropyLoss(), _randn(8, 3), _classes(8, 3))
    yield PairLossOp(nn.MarginRankingLoss(), _randn(8), _randn(8), _signs(8))
    yield TripletLossOp(
        nn.TripletMarginLoss(), _randn(8, 4), _randn(8, 4), _randn(8, 4)
    )
    yield AccuracyOp(_randn(8, 3), _classes(8, 3))


@pytest.fixture(params=_scores(), ids=lambda score: type(score).__name__)
def score(request: pytest.FixtureRequest) -> ScoreOp:
    return request.param


@pytest.fixture(
    params=[nn.MSELoss(), nn.L1Loss(), nn.HuberLoss(), nn.SmoothL1Loss()],
    ids=lambda module: type(module).__name__,
)
def regression_loss(request: pytest.FixtureRequest) -> nn.Module:
    return request.param


def _replace_first_input(score: ScoreOp, op: Op) -> ScoreOp:
    "Rebuild `score` with its first `Op` input replaced, which rechecks the domains."

    fields = dcls.fields(score)
    first = next(f.name for f in fields if isinstance(getattr(score, f.name), Op))
    return dcls.replace(score, **{first: op})


def test_score_is_a_number(score: ScoreOp):
    assert score().shape == ()


def test_non_tensor_input(score: ScoreOp):
    with pytest.raises(OpDomainMismatch):
        _replace_first_input(score, ValueOp(torch.randn(8, 4)))


def test_generic_tensor_input(score: ScoreOp):
    # `TensorDomain` is not specific enough: every first input requires a float tensor.
    with pytest.raises(OpDomainMismatch):
        _replace_first_input(score, TensorOp(torch.randn(8, 4)))


def test_int_input_for_float(score: ScoreOp):
    with pytest.raises(OpDomainMismatch):
        _replace_first_input(score, _ints(torch.randint(0, 3, (8, 4))))


def test_regression_loss_zero_on_target(regression_loss: nn.Module):
    x = torch.randn(8, 4)
    op = RegressionLossOp(regression_loss, _floats(x), _floats(x.clone()))
    tt.assert_close(op(), torch.tensor(0.0))


def test_cross_entropy_uniform():
    logits = _floats(torch.zeros(8, 3))
    op = ClassLossOp(nn.CrossEntropyLoss(), logits, _classes(8, 3))
    tt.assert_close(op(), torch.tensor(math.log(3)))


def test_triplet_zero_when_separated():
    anchor = torch.randn(8, 4)
    op = TripletLossOp(
        nn.TripletMarginLoss(margin=1.0),
        _floats(anchor),
        _floats(anchor.clone()),
        _floats(anchor + 10),
    )
    tt.assert_close(op(), torch.tensor(0.0))


def test_accuracy_half():
    logits = _floats(torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
    target = _ints(torch.tensor([0, 1]))
    tt.assert_close(AccuracyOp(logits, target)(), torch.tensor(0.5))
