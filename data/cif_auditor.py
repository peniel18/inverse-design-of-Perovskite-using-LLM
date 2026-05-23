"""
Step 1 — CIF Auditor
======================
Run this FIRST on your custom CIF folder before any preprocessing.

What it does
------------
  - Reads every .cif file in your directory
  - Tries to parse each one with pymatgen
  - Reports: parse failures, disordered structures, suspiciously small/large
    unit cells, duplicate structures (same composition + volume), and CIFs
    that have no matching label in your CSV (or no CSV yet)
  - Generates a label_template.csv you can fill in if you don't have one yet
  - Generates a audit_report.txt with a full summary

Usage
-----
    # If you already have a labels CSV:
    python cif_auditor.py --cif_dir ./my_cifs --label_csv my_labels.csv

    # If you need to create a labels CSV from scratch:
    python cif_auditor.py --cif_dir ./my_cifs --generate_template

Your label CSV must have AT LEAST these columns:
    filename   — matches your .cif file names (with or without .cif extension)
    Ef         — formation energy in eV/atom  (DFT-calculated)
    Eg         — bandgap in eV                (DFT-calculated)

Example CSV row:
    my_perovskite_001.cif, -3.12, 1.60
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

warnings.filterwarnings("ignore")

from pymatgen.core import Structure, Composition


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def _try_parse(cif_path: str) -> Tuple[Optional[Structure], Optional[str]]:
    """Attempt to parse a CIF. Returns (structure, error_message)."""
    try:
        s = Structure.from_file(cif_path)
        return s, None
    except Exception as e:
        return None, str(e)


def _composition_key(s: Structure) -> str:
    return s.composition.reduced_formula


def audit_cif_directory(
    cif_dir:    str,
    label_csv:  Optional[str] = None,
    out_dir:    str = ".",
) -> Dict:
    """
    Full audit of a CIF directory.

    Returns a dict with keys:
        ok          — list of (filename, structure) that parsed cleanly
        failed      — list of (filename, error)
        disordered  — list of filenames with partial occupancy
        tiny        — list of filenames with < 2 atoms
        large       — list of filenames with > 200 atoms
        duplicates  — list of groups of duplicate filenames
        unlabelled  — list of CIF filenames with no matching label row
    """
    cif_files = sorted([f for f in os.listdir(cif_dir) if f.endswith(".cif")])
    print(f"\nScanning {len(cif_files)} CIF files in '{cif_dir}' ...\n")

    results = {
        "ok":          [],
        "failed":      [],
        "disordered":  [],
        "tiny":        [],
        "large":       [],
        "duplicates":  [],
        "unlabelled":  [],
    }

    # ── Parse every CIF ──────────────────────────────────────────────
    vol_map: Dict[str, List[str]] = defaultdict(list)   # comp_vol → filenames

    for i, fname in enumerate(cif_files):
        path = os.path.join(cif_dir, fname)
        s, err = _try_parse(path)

        if s is None:
            results["failed"].append((fname, err))
            continue

        N = len(s)

        if not s.is_ordered:
            results["disordered"].append(fname)

        if N < 2:
            results["tiny"].append(fname)
        elif N > 200:
            results["large"].append(fname)

        # Duplicate fingerprint: reduced formula + rounded volume
        vol_key = f"{_composition_key(s)}_{round(s.volume, 1)}"
        vol_map[vol_key].append(fname)

        results["ok"].append((fname, s))

        if (i + 1) % 50 == 0:
            print(f"  Parsed {i+1}/{len(cif_files)} ...")

    # ── Find duplicates ───────────────────────────────────────────────
    for key, fnames in vol_map.items():
        if len(fnames) > 1:
            results["duplicates"].append(fnames)

    # ── Cross-check labels ────────────────────────────────────────────
    if label_csv and os.path.exists(label_csv):
        df = pd.read_csv(label_csv)
        labelled_keys = set(
            os.path.splitext(os.path.basename(str(x)))[0]
            for x in df["filename"].tolist()
        )
        for fname, _ in results["ok"]:
            key = os.path.splitext(fname)[0]
            if key not in labelled_keys:
                results["unlabelled"].append(fname)

    return results, cif_files


def print_and_save_report(results: Dict, cif_files: List[str],
                           out_path: str, label_csv: Optional[str]):
    """Write audit report to file and print summary to console."""

    lines = []
    lines.append("=" * 60)
    lines.append("  CIF AUDIT REPORT")
    lines.append("=" * 60)
    lines.append(f"  Total CIF files scanned : {len(cif_files)}")
    lines.append(f"  Successfully parsed     : {len(results['ok'])}")
    lines.append(f"  Parse failures          : {len(results['failed'])}")
    lines.append(f"  Disordered structures   : {len(results['disordered'])}")
    lines.append(f"  Tiny  (< 2 atoms)       : {len(results['tiny'])}")
    lines.append(f"  Large (> 200 atoms)     : {len(results['large'])}")
    lines.append(f"  Duplicate groups        : {len(results['duplicates'])}")
    if label_csv:
        lines.append(f"  CIFs with no label      : {len(results['unlabelled'])}")
    lines.append("")

    if results["failed"]:
        lines.append("── FAILED TO PARSE ─────────────────────────────────────")
        for fname, err in results["failed"]:
            lines.append(f"  {fname}")
            lines.append(f"    Error: {err[:120]}")
        lines.append("")

    if results["disordered"]:
        lines.append("── DISORDERED STRUCTURES ───────────────────────────────")
        lines.append("  (partial occupancy — will be auto-ordered during preprocessing)")
        for fname in results["disordered"]:
            lines.append(f"  {fname}")
        lines.append("")

    if results["duplicates"]:
        lines.append("── DUPLICATE GROUPS (same composition + volume) ────────")
        for group in results["duplicates"]:
            lines.append("  Group: " + ", ".join(group))
        lines.append("")

    if results["unlabelled"]:
        lines.append("── CIFs WITH NO LABEL ──────────────────────────────────")
        for fname in results["unlabelled"]:
            lines.append(f"  {fname}")
        lines.append("")

    report_text = "\n".join(lines)
    print(report_text)

    with open(out_path, "w") as f:
        f.write(report_text)
    print(f"Report saved → {out_path}")


def generate_label_template(results: Dict, out_path: str):
    """
    Generate a CSV template with one row per successfully parsed CIF.
    Fill in the Ef and Eg columns with your DFT values.
    """
    rows = []
    for fname, s in results["ok"]:
        rows.append({
            "filename":    fname,
            "formula":     s.composition.reduced_formula,
            "n_atoms":     len(s),
            "volume_A3":   round(s.volume, 3),
            "Ef":          "",    # ← fill in: formation energy eV/atom
            "Eg":          "",    # ← fill in: bandgap eV
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nLabel template saved → {out_path}")
    print("Fill in the 'Ef' and 'Eg' columns with your DFT-calculated values.")
    print("Then re-run with --label_csv to check for missing labels.\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit a folder of CIF files")
    parser.add_argument("--cif_dir",  required=True)
    parser.add_argument("--label_csv",         default=None,
                        help="Optional: existing CSV to cross-check against CIFs")
    parser.add_argument("--generate_template", action="store_true",
                        help="Write label_template.csv for unlabelled CIFs")
    parser.add_argument("--out_dir",           default=".",
                        help="Where to save audit_report.txt")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results, cif_files = audit_cif_directory(
        args.cif_dir, args.label_csv, args.out_dir)

    print_and_save_report(
        results, cif_files,
        out_path  = os.path.join(args.out_dir, "audit_report.txt"),
        label_csv = args.label_csv,
    )

    if args.generate_template:
        generate_label_template(
            results,
            out_path=os.path.join(args.out_dir, "label_template.csv"),
        )
