# Copyright (c) AIoWay Authors - All Rights Reserved

"Launch github actions jobs in groups."

import argparse
import contextlib as ctxl
import os
import shutil
import subprocess
import sys


@ctxl.contextmanager
def group_by_stage(stage: str):
    print(f"::group::{stage}", flush=True)
    try:
        yield
    finally:
        print(f"::endgroup::", flush=True)


def launch_proc(command: list[str], stage: str) -> int:
    # Set columns to current terminal size - indent.
    env = {**os.environ, **term_size_env()}

    with group_by_stage(stage=stage):
        completed = subprocess.run(
            command, env=env, stdout=sys.stdout, stderr=sys.stdout
        )

    return completed.returncode


def term_size_env() -> dict[str, str]:
    try:
        terminal_size = shutil.get_terminal_size()
    except IOError:
        return {}

    env = {}
    env["COLUMNS"] = str(terminal_size.columns)
    env["LINES"] = str(terminal_size.lines)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args, command = parser.parse_known_args()
    ret_code = launch_proc(command, stage=" ".join(command))
    raise SystemExit(ret_code)
