# Copyright (c) AIoWay Authors - All Rights Reserved

"The states for `Env`s."

import abc


class EnvState[O = object, S = object](abc.ABC):
    """
    `EnvState` is the state for the environment.

    A state supports exposing observation
    """

    @abc.abstractmethod
    def __getstate__(self) -> S:
        raise NotImplementedError

    @abc.abstractmethod
    def __setstate__(self, item: S) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def observ(self) -> O:
        raise NotImplementedError
