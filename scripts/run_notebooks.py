#!/usr/bin/env python3
"""Execute analysis notebooks in numbered order, writing outputs in place."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb"))

def main():
    if not NOTEBOOKS:
        raise SystemExit("No numbered notebooks found")
    kernel = "online-shopper-analysis"
    subprocess.run([sys.executable, "-m", "ipykernel", "install", "--user",
                    "--name", kernel, "--display-name", kernel], cwd=ROOT, check=True)
    for notebook in NOTEBOOKS:
        print(f"Executing {notebook.name}")
        subprocess.run([sys.executable, "-m", "jupyter", "nbconvert",
                        "--to", "notebook", "--execute", "--inplace",
                        "--ExecutePreprocessor.timeout=600",
                        f"--ExecutePreprocessor.kernel_name={kernel}",
                        str(notebook)],
                       cwd=ROOT, check=True)

if __name__ == "__main__":
    main()
