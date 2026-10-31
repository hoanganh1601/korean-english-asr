"""Setup NTFS Junctions on Windows to link data/datasets to data/raw and data/processed.

Equivalent to scripts/setup_data_links.sh for Windows.
"""

from __future__ import annotations

import _winapi
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT / "data" / "datasets"

LINKS = [
    (PROJECT_ROOT / "data/raw/zeroth", DATASETS_ROOT / "zeroth_korean"),
    (PROJECT_ROOT / "data/raw/librispeech", DATASETS_ROOT / "librispeech/LibriSpeech"),
    (PROJECT_ROOT / "data/raw/hike", DATASETS_ROOT / "hike/HiKE-619ebb54-extracted"),
    (
        PROJECT_ROOT / "data/processed/librispeech/dc_corrected",
        DATASETS_ROOT / "librispeech_processed_v1/dc_corrected",
    ),
]


def main():
    print(f"Project root:  {PROJECT_ROOT}")
    print(f"Datasets root: {DATASETS_ROOT}\n")

    for link, target in LINKS:
        if not target.exists():
            print(f"[ERROR] Target directory not found: {target}")
            return 1

        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists():
            print(f"[EXISTS] {link.relative_to(PROJECT_ROOT)}")
        else:
            _winapi.CreateJunction(str(target), str(link))
            print(f"[CREATED] {link.relative_to(PROJECT_ROOT)} -> {target}")

    print("\nAll junctions are in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
