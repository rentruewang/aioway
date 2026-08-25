# Copyright (c) AIoWay Authors - All Rights Reserved

import argparse
import subprocess
import textwrap

parser = argparse.ArgumentParser()
parser.add_argument("--indent", type=str, default="> ")
args, command = parser.parse_known_args()

p = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
assert p.stdout is not None

for line in p.stdout:
    print(textwrap.indent(line, args.indent), end="")

raise SystemExit(p.wait())
