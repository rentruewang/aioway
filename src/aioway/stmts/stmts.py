# Copyright (c) AIoWay Authors - All Rights Reserved

import abc

import loguru as L

__all__ = ["Stmt"]


class Stmt(abc.ABC):
    """
    The base class for statements.

    A statement can modify global state or perform expensive computation.

    It exists for state mutation, when `Expr` is stateless. For instance:

    1. Create or update `nn.Module`s.
    2. Create or update `Dataset`s.
    """

    def __call__(self) -> None:
        L.logger.info("Executing {}", self)
        self.execute()

    @abc.abstractmethod
    def execute(self) -> None:
        raise NotImplementedError
