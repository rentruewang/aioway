# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
import os

import nox


@nox.session
def publish(session: nox.Session):
    "Nox `publish` command. Calls `pdm publish`."
    commands(session).publish()


@nox.session
def build(session: nox.Session):
    "Nox `build` command. Calls `pdm build`."
    commands(session).build()


@nox.session
def pre_commit(session: nox.Session):
    "Runs the pre-commit commands."

    formatting(session)
    typing(session)


@nox.session
def testing(session: nox.Session):
    "Nox `testing` command. Calls `pytest` command. Runs in multiple python versions (if supported)."
    commands(session).test()


@nox.session
def formatting(session: nox.Session):
    "Nox `formatting` command. Calls `autoflake`, `isort`, `black`, in that order."
    autoflake(session)
    isort(session)
    black(session)


@nox.session
def autoflake(session: nox.Session):
    "Nox `autoflake` command. Calls `autoflake` command."
    commands(session).autoflake()


@nox.session
def isort(session: nox.Session):
    "Nox `isort` command. Calls `isort` command."
    commands(session).isort()


@nox.session
def black(session: nox.Session):
    "Nox `black` command. Calls `black` command."
    commands(session).black()


@nox.session
def mypy(session: nox.Session):
    "Nox `mypy` command. Calls `mypy` command."
    commands(session).mypy()


@nox.session
def typing(session: nox.Session):
    "Nox `typing` command. Calls `mypy` command."
    mypy(session)


@functools.cache
def env(session: nox.Session):
    "Global singleton for github."
    return _Environment(session)


@functools.cache
def commands(session: nox.Session):
    "Global singleton for commands."
    return _Commands(session)


@dcls.dataclass(frozen=True)
class _Environment:
    "The manager for setting up github."

    session: nox.Session
    "The nox session to use."

    def __post_init__(self) -> None:
        "Setup environment."

        self._github_cleanup()

        if self.in_github():
            self._run("pdm", "config", "python.use_venv", "true")

    def _github_cleanup(self):

        # Does nothing outside of GitHub Actions.
        if not self.in_github():
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

    @staticmethod
    def in_github() -> bool:
        "Detect whether or not it is running in GitHub Actions."

        print("Checking if we are in GitHub Actions...", end=" ")
        result = os.getenv("GITHUB_ACTIONS") == "true"
        print("Yes" if result else "No")
        return result

    def pdm_build(self):
        self.pdm_install()
        self._run("pdm", "build")

    def pdm_publish(self):
        self.pdm_install()

        # Remove all uncommitted changes s.t. it doesn't mess with builds.
        _ = self._run("git", "reset", "--hard", "HEAD")

        self._run("pdm", "publish")

    def pdm_run(self, *args: str):
        self.pdm_install()
        self._run("pdm", "run", *args)

    def pdm_install(self) -> None:
        # Don't repeatedly reinstall locally.
        if not self.in_github():
            return

        self.session.run_install("pdm", "install", "-G:all")

    def _run(self, *args: str):
        _ = self.session.run_install(*args, external=True)


@dcls.dataclass(frozen=True)
class _Commands:
    session: nox.Session

    def __post_init__(self):
        _ = self.env

    def build(self):
        "`pdm build` command."
        self.env.pdm_build()

    def publish(self):
        "`pdm publish` command."
        self.env.pdm_publish()

    def test(self):
        "`pytest` command."
        self.env.pdm_run("pytest")

    def autoflake(self):
        "`autoflake` command."
        self.env.pdm_run("autoflake", ".")

    def isort(self):
        "`isort` command."
        self.env.pdm_run("isort", ".")

    def black(self):
        "`black` command."
        self.env.pdm_run("black", ".")

    def mypy(self):
        "`mypy` command."
        self.env.pdm_run("mypy", "src")

    @property
    def env(self):
        return env(self.session)
