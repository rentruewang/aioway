# Copyright (c) AIoWay Authors - All Rights Reserved

"The states for `Env`s."

import abc


class EnvState[S = object](abc.ABC):
    """
    `EnvState` is the state for the environment.
    """

    @abc.abstractmethod
    def __getstate__(self) -> S:
        raise NotImplementedError

    @abc.abstractmethod
    def __setstate__(self, item: S) -> None:
        raise NotImplementedError
