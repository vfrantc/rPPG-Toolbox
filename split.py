#!/usr/bin/env python3
"""
Move 10 % of *.mat files from SCAMPS/Train/ to SCAMPS/Test/ and 10 % to SCAMPS/Val/.

• Only touches files that are currently in Train/.
• Uses a fixed RNG seed for reproducibility.
• Prints a concise summary when finished.
"""

import random
import shutil
from pathlib import Path
import sys

# ---- Configuration ---------------------------------------------------------
BASE_DIR = Path("/home/vladimirfrants9/rPPG-Toolbox/data/SCAMPS")
TRAIN = BASE_DIR / "Train"
TEST  = BASE_DIR / "Test"
VAL   = BASE_DIR / "Val"
PERCENT = 0.10          # 10 %
SEED = 42               # change / remove for non-reproducible split
# ----------------------------------------------------------------------------

def main() -> None:
    # Sanity checks
    for d in (TRAIN, TEST, VAL):
        if not d.exists():
            sys.exit(f"✖ Directory not found: {d}")

    # Collect *.mat files currently in Train/
    files = [p for p in TRAIN.glob("*.mat") if p.is_file()]
    if not files:
        sys.exit("✖ No .mat files found in Train/")

    n_total = len(files)
    n_test  = round(n_total * PERCENT)
    n_val   = round(n_total * PERCENT)

    random.seed(SEED)
    test_set = set(random.sample(files, n_test))
    remaining = [p for p in files if p not in test_set]
    val_set = set(random.sample(remaining, n_val))

    # Move files
    for src in test_set:
        shutil.move(str(src), TEST / src.name)
    for src in val_set:
        shutil.move(str(src), VAL / src.name)

    print(f"✔ Moved {len(test_set)} files to {TEST.relative_to(BASE_DIR.parent)}, "
          f"{len(val_set)} to {VAL.relative_to(BASE_DIR.parent)}. "
          f"{n_total - n_test - n_val} files remain in Train/.")

if __name__ == "__main__":
    main()
