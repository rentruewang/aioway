# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from numpy.typing import NDArray
from sklearn import exceptions as skexc
from sklearn.utils import validation as skval

from aioway._ufuncs import AdHocUFunc, UFunc

__all__ = ["SklearnModel", "HasFit", "HasPredict", "HasFitPredict"]


@typing.runtime_checkable
class HasFit(typing.Protocol):
    def fit(self, X, y) -> typing.Self: ...


@typing.runtime_checkable
class HasPredict(typing.Protocol):
    def predict(self, X) -> NDArray: ...


@typing.runtime_checkable
class HasFitPredict(HasFit, HasPredict, typing.Protocol): ...


class SklearnModel[M: HasFitPredict](HasFitPredict):
    """
    `SklearnModel` is a wrapper around Sklearn's API, allowing methods to be used as UFunc.
    """

    def __init__(self, model: M) -> None:
        if not isinstance(model, HasFitPredict):
            raise TypeError(f"{model=} needs to have `.fit` and `.predict`.")

        self._model = model

    @property
    def fit(self) -> UFunc:
        return AdHocUFunc(self._model.fit)

    @property
    def predict(self) -> UFunc:
        return AdHocUFunc(self._model.predict)

    @property
    def model(self) -> M:
        return self._model

    @property
    def is_fit(self) -> bool:
        try:
            skval.check_is_fitted(self.model)
        except skexc.NotFittedError:
            return False
        else:
            return True
