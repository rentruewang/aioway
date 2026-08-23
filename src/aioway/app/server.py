# Copyright (c) AIoWay Authors - All Rights Reserved

import functools
import typing
from collections import abc as cabc

if typing.TYPE_CHECKING:
    import fastapi
    import fastmcp

__all__ = ["serve", "fastapi_app", "fastmcp_app"]


@functools.cache
def fastapi_app() -> fastapi.FastAPI:
    "The public fastapi factory. If importing failed, return `NotImplemented`."
    try:
        return _fastapi_app()
    except ImportError:
        return NotImplemented


@functools.cache
def fastmcp_app() -> fastmcp.FastMCP:
    "The public fastmcp factory. If importing failed, return `NotImplemented`."
    try:
        return _fastmcp_app()
    except ImportError:
        return NotImplemented


def serve[T: cabc.Callable](path: str) -> cabc.Callable[[T], T]:
    def decorator(func: T) -> T:
        if (mcp := fastmcp_app()) is not NotImplemented:
            mcp.tool()(func)

        if (api := fastapi_app()) is not NotImplemented:
            api.get(path)(func)

        return func

    return decorator


def _fastapi_app() -> fastapi.FastAPI:
    import fastapi

    return fastapi.FastAPI()


def _fastmcp_app() -> fastmcp.FastMCP:
    import fastmcp

    return fastmcp.FastMCP()
