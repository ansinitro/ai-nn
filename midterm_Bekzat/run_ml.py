import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ML_ROOT = ROOT / "Labs_notebooks" / "9.3 Machine Learning"


def execute_notebook(path: Path) -> bool:
    cmd = [
        "uv", "run",
        "--with", "jupyter",
        "--with", "numpy",
        "--with", "pandas",
        "--with", "scikit-learn",
        "--with", "matplotlib",
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=600",
        str(path),
    ]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        print(f"[ok] {path.name}")
        return True
    print(f"[failed] {path.name}")
    print(result.stderr)
    return False


def main() -> None:
    notebooks = sorted(ML_ROOT.glob("*/*.ipynb"))
    if not notebooks:
        raise SystemExit(f"No machine learning notebooks found in {ML_ROOT}")

    successes = sum(execute_notebook(path) for path in notebooks)
    print(f"Executed {successes}/{len(notebooks)} machine learning notebooks")


if __name__ == "__main__":
    main()
