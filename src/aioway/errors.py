# Copyright (c) AIoWay Authors - All Rights Reserved

"""User facing high level erros. Subclasses of `AiowayError`."""

import contextlib as ctxl
import types
import typing

__all__ = [
    "AiowayError",
    "FrameworkUnexpected",
    "GitHubTicketFiled",
]


class AiowayError(Exception):
    """
    `AiowayError` is the error thrown by the `aioway` library.
    This captures all the user facing errors that might be raised by `aioway`.
    """

    @ctxl.contextmanager
    @classmethod
    def relay(cls):
        """
        Pass the exception captured on to the next block,
        but re export the exception s.t. it would be a new exception,
        but keeping the original traceback for debugging purposes.

        This method exists because there is a no low level policy in exceptions,
        all user facing exceptions must be subclasses of `AiowayError`.

        This can provide a simple error handling to prevent crashes.
        """

        try:
            yield
        except Exception as e:
            raise cls from e


class FrameworkUnexpected(AiowayError):
    """
    Used when an external framework does not behave as expected.

    Todo:
        Maybe track the usage of what functions automatically,
        by perhaps monkey patching?

        E.g. we can do something like `with trace(module): ...`,
        where `trace` modifies the modules to do tracking.
    """

    def __init__(self, module: types.ModuleType, *reasons: str) -> None:
        self._module = module
        self._reasons = reasons

    def __str__(self) -> str:
        msg = []

        msg.append(f"Problematic module: {str(self._module)}")

        if self._reasons:
            msg.extend(self._reasons)

        return "\n".join(msg)


class GitHubTicketFiled(NotImplementedError):
    """
    The ticket is filed on GitHub, when encountered, show the URL.
    """

    REPO_URL: typing.ClassVar[str] = "https://github.com/rentruewang/aioway"

    def __init__(self, ticket: int = 0, /, *messages: str) -> None:
        super().__init__()

        self._ticket = ticket
        self._msgs = messages

    def __str__(self) -> str:
        msg = []

        if self._ticket:
            msg.append(f"Ticket link: {self.REPO_URL}/issues/{self._ticket}")

        if self._msgs:
            msg.extend(self._msgs)

        return "\n".join(msg)


class UserDefinedFunctionError(AiowayError, ValueError):
    """
    The user provided function is not valid.
    """
