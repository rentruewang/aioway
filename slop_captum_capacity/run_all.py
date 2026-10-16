"""Execute the jupytext notebooks and save them as .ipynb files with their outputs (no extra dependencies).

    python run_all.py                    # every 0*.py notebook in this folder, in order
    python run_all.py 06_*.py 08_*.py    # only these

Each notebook runs in its own Python process (like a fresh kernel). Results go to ./executed/<name>.ipynb;
open them in Jupyter, VS Code or any notebook viewer. Printed text, tables and figures are kept.
"""
import ast
import base64
import glob
import io
import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "executed")


def parse(path):
    """Percent-format .py -> list of (kind, source). The YAML header before the first cell is skipped."""
    cells, kind, buf = [], None, []
    for line in open(path, encoding="utf-8").read().split("\n"):
        if line.startswith("# %%"):
            if kind is not None:
                cells.append((kind, buf))
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        elif kind is not None:
            buf.append(line)
    if kind is not None:
        cells.append((kind, buf))
    out = []
    for kind, lines in cells:
        while lines and not lines[-1].strip():
            lines.pop()
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else l[1:] if l.startswith("#") else l for l in lines]
        if lines:
            out.append((kind, "\n".join(lines)))
    return out


class Outputs:
    """Collects one cell's outputs in order: streams, rich displays, errors."""

    def __init__(self):
        self.items = []

    def stream(self, name, text):
        if not text:
            return
        if self.items and self.items[-1].get("output_type") == "stream" and self.items[-1]["name"] == name:
            self.items[-1]["text"] += text
        else:
            self.items.append({"output_type": "stream", "name": name, "text": text})

    def display(self, data):
        self.items.append({"output_type": "display_data", "data": data, "metadata": {}})


class Tee(io.TextIOBase):
    def __init__(self, outputs, name):
        self.outputs, self.name = outputs, name

    def write(self, text):
        self.outputs.stream(self.name, text)
        return len(text)


def rich(obj):
    data = {"text/plain": repr(obj)}
    if hasattr(obj, "_repr_html_"):
        html = obj._repr_html_()
        if html:
            data["text/html"] = html
    return data


def run_one(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.chdir(os.path.dirname(os.path.abspath(path)))
    sys.path.insert(0, os.getcwd())
    current = Outputs()

    def show(*args, **kwargs):
        for num in plt.get_fignums():
            buf = io.BytesIO()
            plt.figure(num).savefig(buf, format="png", bbox_inches="tight")
            current.display({"image/png": base64.b64encode(buf.getvalue()).decode(), "text/plain": "<Figure>"})
        plt.close("all")

    def display(*objs, **kwargs):
        for obj in objs:
            current.display(rich(obj))

    plt.show = show
    ns = {"__name__": "__main__", "__file__": os.path.abspath(path)}
    cells, count, failed = [], 0, False
    for kind, src in parse(path):
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})
            continue
        count += 1
        current.items = []
        ran = not failed
        if ran:
            old = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = Tee(current, "stdout"), Tee(current, "stderr")
            try:
                tree = ast.parse(src)
                last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                exec(compile(tree, f"<cell {count}>", "exec"), ns)
                if last is not None:
                    value = eval(compile(ast.Expression(last.value), f"<cell {count}>", "eval"), ns)
                    if value is not None:
                        current.items.append({"output_type": "execute_result", "execution_count": count,
                                              "data": rich(value), "metadata": {}})
                show()
            except Exception as exc:  # keep going to write the notebook, but stop executing
                failed = True
                current.items.append({"output_type": "error", "ename": type(exc).__name__, "evalue": str(exc),
                                      "traceback": traceback.format_exc().splitlines()})
            finally:
                sys.stdout, sys.stderr = old
                ns["display"] = display  # `from toy import *` brings its own display; use the capturing one
        cells.append({"cell_type": "code", "execution_count": count if ran else None,
                      "metadata": {}, "outputs": list(current.items), "source": src.splitlines(keepends=True)})
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    os.makedirs(OUT, exist_ok=True)
    target = os.path.join(OUT, os.path.basename(path)[:-3] + ".ipynb")
    with open(target, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    return 1 if failed else 0


def main(args):
    if args[:1] == ["--one"]:
        sys.exit(run_one(args[1]))
    paths = args or sorted(glob.glob(os.path.join(HERE, "0*.py")))
    bad = []
    for path in paths:
        t0 = time.time()
        code = subprocess.call([sys.executable, os.path.abspath(__file__), "--one", os.path.abspath(path)])
        status = "ok" if code == 0 else "FAILED (see the error cell in the notebook)"
        print(f"{os.path.basename(path):34s} {time.time() - t0:5.0f}s  {status}", flush=True)
        if code:
            bad.append(path)
    print(f"notebooks with outputs: {OUT}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
