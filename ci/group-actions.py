# Copyright (c) AIoWay Authors - All Rights Reserved

import argparse
import contextlib as ctxl
import os
import shutil
import subprocess
import sys


@ctxl.contextmanager
def group_by_stage(stage: str):
    print(f"::group::{stage}")
    try:
        yield
    finally:
        print(f"::endgroup::")


def launch_proc_and_print(command: list[str], stage: str):
    # Set columns to current terminal size - indent.
    env = {**os.environ, **term_size_env()}

    with group_by_stage(stage=stage):
        subprocess.run(command, env=env, stdout=sys.stdout)


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
    parser.add_argument("--stage", type=str)
    args, command = parser.parse_known_args()
    launch_proc_and_print(command, stage=" ".join(command))
