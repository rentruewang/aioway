# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn, optim

from aioway._ufuncs import BuiltUFunc, UFunc
from aioway.emits import emit
from aioway.io import Dset
from aioway.relalg import LoaderOpt
from aioway.torch.optim import OptimizerUFunc

__all__ = ["Trainer"]


class Trainer:
    def __init__(self, loss_func: nn.Module) -> None:
        self._loss_func = loss_func
        self._ufunc: UFunc | None = None

    def fit(self, x: Dset, y: Dset, loader_opt: LoaderOpt = LoaderOpt()):
        self._ufunc = next(emit(x.space, y.space))
        assert isinstance(self.ufunc, BuiltUFunc), self.ufunc
        optimizer = OptimizerUFunc(optim.AdamW(self.ufunc.parameters()))

        for batch_x, batch_y in zip(x(loader_opt), y(loader_opt)):
            out = self.ufunc(batch_x)
            loss = self._loss_func(out, batch_y)
            optimizer(loss)
            yield loss

    def predict(self, x: Dset, loader_opt: LoaderOpt = LoaderOpt()):
        for batch_x in x(loader_opt):
            yield self.ufunc(batch_x)

    @property
    def ufunc(self) -> UFunc:
        assert self._ufunc is not None
        return self._ufunc
