"""Small helper used to assemble the Colab notebooks (SETUP + data-loading cells are shared)."""
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
_nb3 = json.load(open(ROOT / "notebooks" / "03_ablations_and_baselines.ipynb"))
_code = [c for c in _nb3["cells"] if c["cell_type"] == "code"]
SETUP = "".join(_code[0]["source"])
_load = "".join(_code[1]["source"])
LOAD = _load


def write(name, cells):
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    nb = {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                      "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(ROOT / "notebooks" / name, "w"), indent=1)
