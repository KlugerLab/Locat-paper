"""
Patches simulations_unimodal_locat01_repro.ipynb to:
  - Use locat-0.1 from LOCAT01_PATH
  - Use GPU 2 (free at time of writing)
  - 1.5x larger axis labels (fontsize=15) and titles (fontsize=18)
  - Larger tick labels (labelsize=13)
  - Shared y-axis [0, 1] on all p-value subplots
Then executes the patched notebook via nbconvert.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

NB_IN = Path(__file__).parent / "simulations_unimodal_locat01_repro.ipynb"
NB_OUT = Path(__file__).parent / "simulations_unimodal_locat01_repro_improved.ipynb"
LOCAT01_PATH = "LOCAT01_PATH"
GPU = "2"


def get_src(cell):
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def set_src(cell, text):
    cell["source"] = text


def patch_plot_cell(src: str) -> str:
    # Add fontsize to ylabel / xlabel / title (single-quoted, no existing fontsize)
    src = re.sub(r"plt\.ylabel\('([^']+)'\)", r"plt.ylabel('\1', fontsize=15)", src)
    src = re.sub(r"plt\.xlabel\('([^']+)'\)", r"plt.xlabel('\1', fontsize=15)", src)
    src = re.sub(r"plt\.title\('([^']+)'\)", r"plt.title('\1', fontsize=18)", src)
    # Before every plt.grid(which='both') inject ylim + tick_params
    src = re.sub(
        r"plt\.grid\(which='both'\)",
        "plt.ylim(0, 1)\nplt.tick_params(labelsize=13)\nplt.grid(which='both')",
        src,
    )
    return src


def main():
    with open(NB_IN) as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        s = get_src(cell)

        # Fix locat path
        if "LOCAT01_PATH" in s:
            s = s.replace("'LOCAT01_PATH'", f"'{LOCAT01_PATH}'")

        # Fix GPU selection
        if "CUDA_VISIBLE_DEVICES" in s:
            s = re.sub(
                r"os\.environ\['CUDA_VISIBLE_DEVICES'\]\s*=\s*'[^']+'",
                f"os.environ['CUDA_VISIBLE_DEVICES'] = '{GPU}'",
                s,
            )

        # Patch plot cells (each has figsize= and subplot(221))
        if "plt.figure(figsize=" in s and "plt.subplot(221)" in s:
            s = patch_plot_cell(s)

        set_src(cell, s)

    with open(NB_OUT, "w") as f:
        json.dump(nb, f, indent=1)

    print(f"Patched notebook written to: {NB_OUT}")

    python = "LOCAT_PYTHON"
    cmd = [
        python, "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=3600",
        "--inplace",
        str(NB_OUT),
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
