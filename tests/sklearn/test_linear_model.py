# Copyright (c) AIoWay Authors - All Rights Reserved

import inspect

import pytest
from sklearn import linear_model

from aioway.sklearn import SklearnModel


@pytest.fixture
def linear_regr():
    return SklearnModel(linear_model.LinearRegression())


@pytest.fixture
def linear_clf():
    return SklearnModel(linear_model.RidgeClassifier())


@pytest.fixture(params=[linear_regr.name, linear_clf.name])
def linear(request: pytest.FixtureRequest):
    return request.getfixturevalue(request.param)


def test_linear_signature(linear: SklearnModel):
    assert inspect.signature(linear.fit) == inspect.signature(linear.model.fit)
    assert inspect.signature(linear.predict) == inspect.signature(linear.model.predict)


def test_linear_fit(linear: SklearnModel):
    assert not linear.is_fit
