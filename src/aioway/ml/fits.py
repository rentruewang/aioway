# Copyright (c) AIoWay Authors - All Rights Reserved

"The generic interface for `Fit`."

import abc

__all__ = ["Fit"]


class Fit(abc.ABC):
    @abc.abstractmethod
    def __call__(self, module, data) -> None:
        raise NotImplementedError
