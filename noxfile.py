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
def github(session: nox.Session):
    "Global singleton for github."
    return _Github(session)


@functools.cache
def pdm(session: nox.Session):
    "Global singleton for pdm."
    return _Pdm(session)


@functools.cache
def commands(session: nox.Session):
    "Global singleton for commands."
    return _Commands(session)


@dcls.dataclass(frozen=True)
class _Github:
    "The manager for setting up github."

    session: nox.Session
    "The nox session to use."

    @functools.cache
    def setup(self) -> None:
        "The shared entrypoint to GitHub Actions scripts"

        # Does nothing outside of GitHub Actions.
        if not self.active():
            return

        self._remove_unwanted_files()
        self._log_storage_usage()

    def _run(self, *args: str):
        self.session.run_install(*args, external=True)

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
    def active() -> bool:
        "Detect whether or not it is running in GitHub Actions."

        print("Checking if we are in GitHub Actions...", end=" ")
        result = os.getenv("GITHUB_ACTIONS") == "true"
        print("Yes" if result else "No")
        return result


@dcls.dataclass(frozen=True)
class _Pdm:
    session: nox.Session

    def __post_init__(self):
        github(self.session).setup()

        if _is_remote(self.session):
            self._run("pdm", "config", "python.use_venv", "true")

    def sync(self) -> None:
        self._sync_or_install("sync")

    def install(self):
        self._sync_or_install("install")

    def build(self):
        self.install()
        self._run("pdm", "build")

    def publish(self):
        self.install()

        # Remove all uncommitted changes s.t. it doesn't mess with builds.
        _ = self.session.run("git", "reset", "--hard", "HEAD")

        self._run("pdm", "publish")

    def run(self, *args: str):
        self.sync()
        self._run("pdm", "run", *args)

    def _sync_or_install(self, mode: str) -> None:
        # Don't repeatedly reinstall locally.
        if not _is_remote(self.session):
            return

        self.session.run_install("pdm", mode, "-G:all")

    def _run(self, *args: str):
        self.session.run(*args, external=True)


@dcls.dataclass(frozen=True)
class _Commands:
    session: nox.Session

    def __post_init__(self):
        github(self.session).setup()

    def build(self):
        "`pdm build` command."
        self.pdm.build()

    def publish(self):
        "`pdm publish` command."
        self.pdm.publish()

    def test(self):
        "`pytest` command."
        self.pdm.run("pytest")

    def autoflake(self):
        "`autoflake` command."
        self.pdm.run("autoflake", ".")

    def isort(self):
        "`isort` command."
        self.pdm.run("isort", ".")

    def black(self):
        "`black` command."
        self.pdm.run("black", ".")

    def mypy(self):
        "`mypy` command."
        self.pdm.run("mypy", "src")

    @property
    def pdm(self):
        return pdm(self.session)


def _is_remote(session: nox.Session):
    return github(session).active()
