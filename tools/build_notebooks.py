"""Convert percent-format scripts in tools/nb_src/ into executed .ipynb notebooks.

Usage:  python tools/build_notebooks.py [name ...]

Cells are separated by lines starting with ``# %%``; a ``# %% [markdown]`` cell
has each line prefixed by ``# `` which is stripped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tools" / "nb_src"
OUT = ROOT / "notebooks"


def parse(text: str) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    cells: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if line.startswith("# %%"):
            kind = "markdown" if "[markdown]" in line else "code"
            cells.append((kind, []))
        elif cells:
            cells[-1][1].append(line)
    for kind, lines in cells:
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        if kind == "markdown":
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines)
            nb.cells.append(nbformat.v4.new_markdown_cell(body))
        else:
            nb.cells.append(nbformat.v4.new_code_cell("\n".join(lines)))
    return nb


def build(path: Path) -> None:
    nb = parse(path.read_text(encoding="utf-8"))
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(OUT)}})
    out = OUT / (path.stem + ".ipynb")
    nbformat.write(nb, out)
    print(f"built {out.relative_to(ROOT)}")


if __name__ == "__main__":
    names = sys.argv[1:]
    files = sorted(SRC.glob("*.py")) if not names else [SRC / f"{n}.py" for n in names]
    for f in files:
        build(f)
