# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from aioway._iters import Iter
from aioway._ufuncs import UFunc
from aioway.io import Env


class Trainer(Iter):
    """
    The trainer iterator. It is responsible for iterating over the APIs.
    """

    def __init__(self, env: Env, ufunc: UFunc) -> None:
        self._env = env
        self._ufunc = ufunc

    @typing.override
    def iterate(self):
        env_iter = self.env.generator()
        action: typing.Any = None

        while True:
            # First iteration, `.send(None)` is equal to `next`.
            observation = env_iter.send(action)
            action = self.ufunc(observation)

    @property
    def env(self) -> Env:
        return self._env

    @property
    def ufunc(self):
        return self._ufunc
