# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import functools
import os
import pathlib
import subprocess as sp
import sys
from collections import abc as cabc

import nox

ROOT = pathlib.Path(__file__).parent
"The project root."

VENV = os.getenv("VIRTUAL_ENV")
"The venv folder if we are in venv."

_session: nox.Session | None = None
"The global session to simplfy code."

_setup_is_done: bool = False
"If the `setup_env` is called."


@ctxl.contextmanager
def enter_session(session: nox.Session):
    global _session

    prev, _session = _session, session
    try:
        yield session
    finally:
        _session = prev


def _current_session() -> nox.Session:
    assert _session is not None
    return _session


def run(*cmd: str) -> None:
    _ = _current_session().run_always(*cmd, external=True)


def nox_cmd(func: cabc.Callable[[], None]) -> cabc.Callable[[], None]:
    """
    The command that wraps a `nox.Session`,
    allowing you to get the current session in a global function `_current_session()`.
    """

    @functools.wraps(func)
    def wrapper(session: nox.Session):
        with enter_session(session):
            func()

    _ = nox.session(wrapper)

    return func


@nox_cmd
def setup():
    "Check if `nox` can be run (side effect will cleanup github)."

    if in_github():
        _github_cleanup()

    _install_ffmpeg()

    run("df", "-h")


@nox_cmd
def publish():
    "Calls `pdm publish`."
    pdm_publish()


@nox_cmd
def build():
    "Calls `pdm build`."
    pdm_build()


@nox_cmd
def test():
    "Calls `pytest *posargs`."
    pdm_run("pytest", *_current_session().posargs)


@nox_cmd
def format():
    "Calls `autoflake`, `isort`, `black`, in that order."
    autoflake()
    isort()
    black()


@nox_cmd
def format_check():
    "Calls `autoflake`, `isort`, `black`, in that order."
    autoflake_check()
    isort_check()
    black_check()


@nox_cmd
def autoflake():
    "Calls `autoflake`."
    pdm_run("autoflake", ".")


@nox_cmd
def autoflake_check():
    "Calls `autoflake --check`."
    pdm_run("autoflake", "--check", ".")


@nox_cmd
def isort():
    "Calls `isort`."
    pdm_run("isort", ".")


@nox_cmd
def isort_check():
    "Calls `isort --check`."
    pdm_run("isort", "--check", ".")


@nox_cmd
def black():
    "Calls `black`."
    pdm_run("black", ".")


@nox_cmd
def black_check():
    "Calls `black --check`."
    pdm_run("black", "--check", ".")


@nox_cmd
def type():
    "Calls `mypy`. Installs necessary type stubs."
    pdm_run("mypy", "--install-types", "--non-interactive", "src")


def _github_cleanup():
    # Does nothing outside of GitHub Actions.
    if not in_github():
        return

    _remove_unwanted_files()
    _log_storage_usage()


def _remove_unwanted_files() -> None:
    "Remove the files GitHub Actions pre-installed."

    print("Removing files we did not ask for...")

    for folder in [
        "/usr/local/lib/android",
        "/usr/share/dotnet",
        "/usr/local/.ghcup",
    ]:
        run("sudo", "rm", "-rf", folder)

    run("docker", "system", "prune", "-af", "--volumes")


def _log_storage_usage() -> None:
    "Log how much usage is currently being used by GitHub Actions."
    print("Investigating how much storage is used in GitHub Actions...")

    run("df", "-h")


def _install_ffmpeg() -> None:
    if ffmpeg_is_installed():
        return

    # Install since it's not installed yet.
    match sys.platform:
        case "darwin":
            run("brew", "install", "ffmpeg")
        case "linux":
            run("sudo", "apt-get", "update")
            run("sudo", "apt-get", "install", "-y", "ffmpeg")
        case _:
            raise RuntimeError(f"Platform {sys.platform} is not supported yet.")


def pdm_build():
    pdm_update_deps()
    run("pdm", "build")


def pdm_publish():
    # Use install + git reset for maximum flexibility.
    pdm_update_deps("install")

    # Remove all uncommitted changes s.t. it doesn't mess with builds.
    if in_github():
        run("git", "reset", "--hard", "HEAD")

    run("pdm", "publish")


def pdm_run(*args: str):
    pdm_update_deps()
    run("pdm", "run", *args)


def pdm_update_deps(command: str = "sync") -> None:
    # Don't repeatedly reinstall locally.
    if not in_github():
        return

    run("pdm", command, "-G:all")


def checking_if(condition: str):
    def decorator[**P](function: cabc.Callable[P, bool]) -> cabc.Callable[P, bool]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> bool:
            print(f"Checking if {condition}...", end=" ")
            answer = function(*args, **kwargs)
            print("Yes" if answer else "No")
            return answer

        return wrapper

    return decorator


@checking_if("we are in GitHub Actions")
def in_github() -> bool:
    "Detect whether or not it is running in GitHub Actions."

    return os.getenv("GITHUB_ACTIONS") == "true"


@checking_if("ffmpeg is installed")
def ffmpeg_is_installed():
    try:
        result = sp.run(["ffmpeg", "-version"], stdout=sp.PIPE, stderr=sp.PIPE)
        return result.returncode == 0
    except FileNotFoundError:
        return False
