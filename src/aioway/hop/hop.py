# Copyright (c) AIoWay Authors - All Rights Reserved

"The operator base class."

import abc

__all__ = []


class Hop(abc.ABC):
    """
    `Op` stands for [h]igh level [op]erator, or [h]igh level [o]peration [p]review.
    It is essentailly an unevaluated expression that supports inspection.
    """

    @abc.abstractmethod
    def do(self) -> object:
        raise NotImplementedError
