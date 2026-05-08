# Copyright (c) AIoWay Authors - All Rights Reserved

"Turning on logging globally for a duration."

import contextlib as ctxl
import logging
import typing

from rich import logging as rlogging

__all__ = ["enable_log", "enable_rich_log"]


@ctxl.contextmanager
def enable_log(level: str | int, /, *handlers: logging.Handler):
    """
    Enable logging for the duration of the block package wide.
    """

    logger = logging.getLogger("aioway")

    before_level = logger.level
    before_handlers = list(logger.handlers)

    for handler in handlers:
        logger.addHandler(handler)
        logger.setLevel(level)

    try:
        yield logger
    finally:
        logger.setLevel(before_level)
        logger.handlers = before_handlers


@ctxl.contextmanager
def enable_rich_log(level: str | int, /):
    """
    Enable logging for the duration of the block with rich handlers.
    """

    with enable_log(level, rlogging.RichHandler(show_path=False)) as logger:
        yield logger


type LoggingLevel = typing.Literal[
    "NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", 0, 10, 20, 30, 40, 50
]
"The accepted logging levels. Same as `logging` library."
