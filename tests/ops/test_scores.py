# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import math

import pytest
import torch
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


def _floats(*shape: int) -> TensorOp:
    return TensorOp(torch.randn(*shape))


def _bits(*shape: int) -> TensorOp:
    return TensorOp(torch.randint(0, 2, shape).float())


def _signs(*shape: int) -> TensorOp:
    return TensorOp(torch.randint(0, 2, shape).float() * 2 - 1)


def _classes(size: int, num_classes: int) -> TensorOp:
    return TensorOp(torch.randint(0, num_classes, (size,)))


def _scores():
    yield RegressionLossOp(nn.MSELoss(), _floats(8, 4), _floats(8, 4))
    yield BinaryLossOp(nn.BCEWithLogitsLoss(), _floats(8, 4), _bits(8, 4))
    yield ClassLossOp(nn.CrossEntropyLoss(), _floats(8, 3), _classes(8, 3))
    yield PairLossOp(nn.MarginRankingLoss(), _floats(8), _floats(8), _signs(8))
    yield TripletLossOp(
        nn.TripletMarginLoss(), _floats(8, 4), _floats(8, 4), _floats(8, 4)
    )
    yield AccuracyOp(_floats(8, 3), _classes(8, 3))


@pytest.fixture(params=_scores(), ids=lambda score: type(score).__name__)
def score(request: pytest.FixtureRequest) -> ScoreOp:
    return request.param


@pytest.fixture(
    params=[nn.MSELoss(), nn.L1Loss(), nn.HuberLoss(), nn.SmoothL1Loss()],
    ids=lambda module: type(module).__name__,
)
def regression_loss(request: pytest.FixtureRequest) -> nn.Module:
    return request.param


def test_score_is_a_number(score: ScoreOp):
    assert score().shape == ()


def test_non_tensor_input(score: ScoreOp):
    first = next(
        f.name for f in dcls.fields(score) if isinstance(getattr(score, f.name), Op)
    )

    with pytest.raises(OpDomainMismatch):
        dcls.replace(score, **{first: ValueOp(torch.randn(8, 4))})


def test_regression_loss_zero_on_target(regression_loss: nn.Module):
    x = torch.randn(8, 4)
    op = RegressionLossOp(regression_loss, TensorOp(x), TensorOp(x.clone()))
    tt.assert_close(op(), torch.tensor(0.0))


def test_cross_entropy_uniform():
    op = ClassLossOp(nn.CrossEntropyLoss(), TensorOp(torch.zeros(8, 3)), _classes(8, 3))
    tt.assert_close(op(), torch.tensor(math.log(3)))


def test_triplet_zero_when_separated():
    anchor = torch.randn(8, 4)
    op = TripletLossOp(
        nn.TripletMarginLoss(margin=1.0),
        TensorOp(anchor),
        TensorOp(anchor.clone()),
        TensorOp(anchor + 10),
    )
    tt.assert_close(op(), torch.tensor(0.0))


def test_accuracy_half():
    logits = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    target = torch.tensor([0, 1])
    op = AccuracyOp(TensorOp(logits), TensorOp(target))
    tt.assert_close(op(), torch.tensor(0.5))
