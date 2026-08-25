# Copyright (c) AIoWay Authors - All Rights Reserved

import argparse
import os
import subprocess
import textwrap

INDENT = "    | "
COMMAND = "$ "


def box(text: str) -> str:
    """
    Use a box to surround the (maybe multline) text.
    """

    text = COMMAND + text
    lines = text.splitlines()
    max_len = max(len(l) for l in lines)

    top_bottom = "+-" + "-" * max_len + "-+"

    string_builder: list[str] = []
    string_builder.append(top_bottom)

    for l in lines:
        string_builder.append("| " + l + " " * (max_len - len(l)) + " |")

    string_builder.append(top_bottom)
    return "\n".join(string_builder)


def launch_proc_and_print(command: list[str], indent: str) -> int:
    # Set columns to current terminal size - indent.
    env = os.environ.copy()
    env["COLUMNS"] = str(os.get_terminal_size().columns - len(indent))
    env["LINES"] = str(os.get_terminal_size().lines)

    p = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, env=env)
    assert p.stdout is not None

    for line in p.stdout:
        if not line:
            continue

        print(textwrap.indent(line, indent).rstrip())
    print()
    return p.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indent", type=str, default=INDENT)
    args, command = parser.parse_known_args()

    print(box(" ".join(command)))
    exit_code = launch_proc_and_print(command, args.indent)
    raise SystemExit(exit_code)
