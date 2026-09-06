# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway.rewrites import Matcher
import pytest
from torch import nn


@pytest.fixture
def linear():
    return nn.Linear(3, 3)


@pytest.fixture
def sequential(linear: nn.Module):
    return nn.Sequential(linear, linear, linear)


def _linear_to_identity(module: nn.Linear) -> nn.Identity:
    return nn.Identity()


@pytest.fixture
def linear_matcher():
    matcher = Matcher()
    matcher.register(_linear_to_identity)
    return matcher


def test_linear_match_linear(linear_matcher: Matcher, linear: nn.Linear):
    out = linear_matcher(linear)
    assert isinstance(out, nn.Identity)


def test_linear_match_sequential(linear_matcher: Matcher, sequential: nn.Sequential):
    out = linear_matcher(sequential)
    assert out is NotImplemented
