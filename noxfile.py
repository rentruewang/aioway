# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
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


@nox.session
def publish(session: nox.Session):
    "Nox `publish` command. Calls `pdm publish`."
    env(session).pdm_publish()


@nox.session
def build(session: nox.Session):
    "Nox `build` command. Calls `pdm build`."
    env(session).pdm_build()


@nox.session
def test(session: nox.Session):
    "Nox `testing` command. Calls `pytest` command. Runs in multiple python versions (if supported)."
    env(session).pdm_run("pytest", *session.posargs)


@nox.session
def format(session: nox.Session):
    "Nox `formatting` command. Calls `autoflake`, `isort`, `black`, in that order."
    autoflake(session)
    isort(session)
    black(session)
    nb_clean(session)


@nox.session
def format_check(session: nox.Session):
    "Nox `formatting` command. Calls `autoflake`, `isort`, `black`, in that order."
    autoflake_check(session)
    isort_check(session)
    black_check(session)
    nb_check(session)


@nox.session
def nb_clean(session: nox.Session):
    "Call `nb-clean clean`."
    env(session).pdm_run("nb-clean", "clean", "notebooks")


@nox.session
def nb_check(session: nox.Session):
    "Call `nb-clean check`."
    env(session).pdm_run("nb-clean", "check", "notebooks")


@nox.session
def autoflake(session: nox.Session):
    "Nox `autoflake` command. Calls `autoflake` command."
    env(session).pdm_run("autoflake", ".")


@nox.session
def autoflake_check(session: nox.Session):
    "Nox `autoflake` command. Calls `autoflake` command."
    env(session).pdm_run("autoflake", "--check", ".")


@nox.session
def isort(session: nox.Session):
    "Nox `isort` command. Calls `isort` command."
    env(session).pdm_run("isort", ".")


@nox.session
def isort_check(session: nox.Session):
    "Nox `isort` command. Calls `isort` command."
    env(session).pdm_run("isort", "--check", ".")


@nox.session
def black(session: nox.Session):
    "Nox `black` command. Calls `black` command."
    env(session).pdm_run("black", ".")


@nox.session
def black_check(session: nox.Session):
    "Nox `black` command. Calls `black` command."
    env(session).pdm_run("black", "--check", ".")


@nox.session
def type(session: nox.Session):
    "Nox `type` command. Calls `mypy` command."
    env(session).pdm_run("mypy", "--install-types", "--non-interactive", "src")


@functools.cache
def env(session: nox.Session):
    "Global singleton for github."
    return _Environment(session)


@dcls.dataclass(frozen=True)
class _Environment:
    "The manager for setting up github."

    session: nox.Session
    "The nox session to use."

    def __post_init__(self) -> None:
        "Setup environment."

        self._github_cleanup()

        if in_github():
            self._run("pdm", "config", "python.use_venv", "true")

        self._install_ffmpeg()

    def _github_cleanup(self):

        # Does nothing outside of GitHub Actions.
        if not in_github():
            return

        self._remove_unwanted_files()
        self._log_storage_usage()

    def _remove_unwanted_files(self) -> None:
        "Remove the files GitHub Actions pre-installed."

        print("Removing files we did not ask for...")

        for folder in [
            "/usr/local/lib/android",
            "/usr/share/dotnet",
            "/usr/local/.ghcup",
        ]:
            self._run("sudo", "rm", "-rf", folder)

        self._run("docker", "system", "prune", "-af", "--volumes")

    def _log_storage_usage(self) -> None:
        "Log how much usage is currently being used by GitHub Actions."
        print("Investigating how much storage is used in GitHub Actions...")

        self._run("df", "-h")

    def _install_ffmpeg(self) -> None:
        if ffmpeg_is_installed():
            return

        # Install since it's not installed yet.
        match sys.platform:
            case "darwin":
                self._run("brew", "install", "ffmpeg")
            case "linux":
                self._run("sudo", "apt-get", "install", "ffmpeg")
            case _:
                raise RuntimeError(f"Platform {sys.platform} is not supported yet.")

    def pdm_build(self):
        self.pdm_update_deps()
        self._run("pdm", "build")

    def pdm_publish(self):
        self.pdm_update_deps("install")

        # Remove all uncommitted changes s.t. it doesn't mess with builds.
        if in_github():
            _ = self._run("git", "reset", "--hard", "HEAD")

        self._run("pdm", "publish")

    def pdm_run(self, *args: str):
        self.pdm_update_deps()
        self._run("pdm", "run", *args)

    def pdm_update_deps(self, command: str = "sync") -> None:
        # Don't repeatedly reinstall locally.
        if not in_github():
            return

        self.session.run_install("pdm", command, "-G:all")

    def _run(self, *args: str):
        _ = self.session.run_install(*args, external=True)


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
