from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_ROOT = ROOT / "Labs_notebooks"

REQUIRED_FILES = [
    "README.md",
    "Report_9.2_Python.md",
    "Report_9.3_Machine_Learning.md",
    "Report_9.4_Deep_Learning.md",
    "huawei_midterm.tex",
    "huawei_midterm.pdf",
]


def main() -> None:
    missing = [name for name in REQUIRED_FILES if not (NOTEBOOK_ROOT / name).exists()]
    notebooks = sorted(NOTEBOOK_ROOT.glob("9.*/*/*.ipynb"))
    assets = sorted((NOTEBOOK_ROOT / "assets").glob("*.png"))

    for notebook in notebooks:
        json.loads(notebook.read_text(encoding="utf-8"))

    if missing:
        raise SystemExit(f"Missing required files: {missing}")
    if len(notebooks) != 18:
        raise SystemExit(f"Expected 18 notebooks, found {len(notebooks)}")
    if len(assets) < 8:
        raise SystemExit(f"Expected at least 8 assets, found {len(assets)}")

    print(f"Verified {len(notebooks)} notebooks, {len(assets)} assets, and {len(REQUIRED_FILES)} report files")


if __name__ == "__main__":
    main()
