# Copyright (c) AIoWay Authors - All Rights Reserved

import argparse
import os
import pty
import subprocess
import textwrap

INDENT = "    | "
COMMAND = "$ "


def panel(text: str) -> str:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indent", type=str, default=INDENT)
    args, command = parser.parse_known_args()

    print(panel(" ".join(command)))

    # Using PTY to force color.
    master, slave = pty.openpty()

    p = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=slave,
        stderr=slave,
        close_fds=True,
    )

    os.close(slave)

    with os.fdopen(master, "r", encoding="utf-8", errors="replace") as out:
        # Render line by line.
        for line in out:
            print(textwrap.indent(line.rstrip(), args.indent))
    print()

    raise SystemExit(p.wait())
