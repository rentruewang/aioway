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
    assert "\n" not in text

    top_bottom = "+-" + "-" * len(text) + "-+"
    middle = "| " + text + " |"
    return top_bottom + "\n" + middle + "\n" + top_bottom


print(box(" ".join(command)))

p = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
assert p.stdout is not None

for line in p.stdout:
    if not line:
        continue

    print(textwrap.indent(line, args.indent).rstrip())

raise SystemExit(p.wait())
