# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import functools
import os
import pathlib
import shutil
import sys
from collections import abc as cabc

import nox
from nox import command as ncmd

ROOT = pathlib.Path(__file__).parent
"The project root."

VENV = os.getenv("VIRTUAL_ENV")
"The venv folder if we are in venv."

PUBLIC = ROOT / "public"
"The deployement folder path."

DOCS = ROOT / "docs"
"The path for root documentations."

INSTALL_TIMEOUT = "7m"
"Allow 5 minutes for timeouts."

INSTALL_RETRIES = 4
"Allow 3 times for install timeouts."

_session: nox.Session | None = None
"The global session to simplfy code."


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

    _ = nox.session(reuse_venv=True)(wrapper)

    return func


@nox_cmd
def setup():
    "Check if `nox` can be run (side effect will cleanup github)."

    _github_cleanup()

    # Enable the use of `timeouts`.
    _install_coreutils()

    _install_ffmpeg()
    _install_pdm()


@nox_cmd
def install():
    "Perform installation in the environment."
    pdm_update_deps("install")


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


@nox_cmd
def docs():
    "Build the documents, move to 'public' folder."

    PUBLIC.mkdir(exist_ok=True)
    if not (ignore := PUBLIC / ".gitignore").exists():
        ignore.write_text("*")

    sphinx()
    shutil.copytree(DOCS, PUBLIC, dirs_exist_ok=True)
    shutil.copytree(DOCS / "build" / "html", PUBLIC / "api", dirs_exist_ok=True)


@nox_cmd
def sphinx():
    "Run the sphinx documentation build site"

    pdm_run("make", "-C", str(DOCS), "html")


def _github_cleanup():
    # Does nothing outside of GitHub Actions.
    if not _running_in_github():
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


def _install_coreutils() -> None:
    if _timeout_is_installed():
        return

    if sys.platform != "darwin":
        return

    run("brew", "install", "coreutils")


def _reattempt_install(function: cabc.Callable[[], None]):
    "Retry for a certain number of times."

    for retry in range(INSTALL_RETRIES):
        try:
            function()
            return
        except ncmd.CommandFailed:
            print(f"Retry #{retry} failed.")

            # Terminate if fail.
            if retry == INSTALL_RETRIES - 1:
                raise
            else:
                continue


def _install_pdm() -> None:
    if _pdm_is_installed():
        return

    _reattempt_install(_do_pdm_install)


def _do_pdm_install():
    run(*_timeout(), "pipx", "install", "pdm")


def _install_ffmpeg() -> None:
    if _ffmpeg_is_installed():
        return

    _reattempt_install(_do_ffmpeg_install)


def _do_ffmpeg_install():
    # Install since it's not installed yet.
    match sys.platform:
        case "darwin":
            run(*_timeout(), "brew", "install", "ffmpeg")
        case "linux":
            flags = [
                "-o",
                "Acquire::Languages=none",
                "-o",
                "Acquire::ForceIPv4=true",
                "-o",
                "Acquire::http::Pipeline-Depth=0",
            ]

            run("sudo", *_timeout(), "apt-get", "update", *flags)
            run("sudo", *_timeout(), "apt-get", "install", "-y", "ffmpeg")
        case _:
            raise RuntimeError(f"Platform {sys.platform} is not supported yet.")


def pdm_build():
    pdm_update_deps()
    run("pdm", "build")


def pdm_publish():
    # Use install + git reset for maximum flexibility.
    pdm_update_deps("install")

    # Remove all uncommitted changes s.t. it doesn't mess with builds.
    if _running_in_github():
        run("git", "reset", "--hard", "HEAD")

    run("pdm", "publish")


def pdm_run(*args: str):
    pdm_update_deps()
    run("pdm", "run", *args)


def pdm_update_deps(command: str = "sync") -> None:
    # Don't repeatedly reinstall locally.
    if not _running_in_github():
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
def _running_in_github() -> bool:
    "Detect whether or not it is running in GitHub Actions."

    return os.getenv("GITHUB_ACTIONS") == "true"


@checking_if("ffmpeg is installed")
def _ffmpeg_is_installed():
    return _run_success("ffmpeg", "-version")


@checking_if("timeout is installed")
def _timeout_is_installed():
    return _run_success("timeout", "--version")


@checking_if("pdm is installed")
def _pdm_is_installed():
    return _run_success("pdm", "--version")


def _get_env():
    env = os.environ.copy()

    for key, val in _current_session().env.items():
        if val is None:
            continue
        env[key] = val

    return env


def _run_success(*cmd):
    try:
        run(*cmd)
    except ncmd.CommandFailed:
        return False
    else:
        return True


@ctxl.contextmanager
def chdir(to: os.PathLike) -> cabc.Generator[os.PathLike]:
    to = pathlib.Path(to)
    assert to.exists()
    before = os.getcwd()
    os.chdir(to)

    try:
        yield to
    finally:
        os.chdir(before)


def _timeout() -> tuple[str, ...]:
    return "timeout", INSTALL_TIMEOUT
