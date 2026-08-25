# Copyright (c) AIoWay Authors - All Rights Reserved

import argparse
import subprocess
import textwrap

INDENT = "    | "
parser = argparse.ArgumentParser()
parser.add_argument("--indent", type=str, default=INDENT)
args, command = parser.parse_known_args()


def box(text: str):
    text = "> " + text
    lines = text.splitlines()
    max_len = max(len(l) for l in lines)

    top_bottom = "+-" + "-" * max_len + "-+"

    string_builder: list[str] = []
    string_builder.append(top_bottom)

    for l in lines:
        string_builder.append("| " + l + " " * (max_len - len(l)) + " |")

    string_builder.append(top_bottom)
    return "\n".join(string_builder)


print(box(" ".join(command)))

p = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
assert p.stdout is not None

for line in p.stdout:
    if not line:
        continue

    print(textwrap.indent(line, args.indent).rstrip())

raise SystemExit(p.wait())
