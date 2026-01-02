"""
validate_csv.py
By: Natiq Ghafoor

CSV validator for rep output files.

This script validates rep CSVs produced by the project. It supports two common schemas:

1) Reps CSV (from main.py)
   rep_index,start_idx,end_idx,duration,label,reason,(rom optional)

2) Aggregated/plot-ready CSV (older/alternative)
   rep_index,start_frame,end_frame,rep_duration,label,fail_reason,(rom optional)

Checks performed:
- Required columns exist (based on detected schema)
- Required columns do not contain NaN
- Labels are only "pass" or "fail"
- Frame indices are ordered correctly
- rep_index values are unique
- ROM is not negative (0 is allowed for partial/incomplete segments)
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import pandas as pd


def _detect_schema(df: pd.DataFrame) -> Tuple[str, List[str]]:
    """
    Detects which CSV schema is present and returns:
      (schema_name, required_columns)
    """
    cols = set(df.columns)

    reps_csv = ["rep_index", "start_idx", "end_idx", "duration", "label", "reason"]
    aggregated_csv = ["rep_index", "start_frame", "end_frame", "rep_duration", "label", "fail_reason"]

    if all(c in cols for c in reps_csv):
        return "reps_csv", reps_csv

    if all(c in cols for c in aggregated_csv):
        return "aggregated_csv", aggregated_csv

    # Unknown schema: keep the minimum required fields so error output is still useful.
    return "unknown", ["rep_index", "label"]


def validate_csv(csv_path: str) -> None:
    """
    Loads a CSV file and prints validation results to stdout.
    """
    df = pd.read_csv(csv_path)
    errors: List[str] = []

    schema_name, required_cols = _detect_schema(df)

    # Required column presence
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    # Required column null check (avoid flagging optional columns)
    for col in required_cols:
        if col in df.columns and df[col].isnull().any():
            errors.append(f"Missing (NaN) values in required column: {col}")

    # Label check
    if "label" in df.columns:
        labels = df["label"].astype(str).str.lower()
        if not labels.isin(["pass", "fail"]).all():
            errors.append("Invalid values in 'label' column (expected 'pass' or 'fail')")

    # Frame order checks (schema-specific)
    if schema_name == "reps_csv":
        if "start_idx" in df.columns and "end_idx" in df.columns:
            if (pd.to_numeric(df["start_idx"], errors="coerce") > pd.to_numeric(df["end_idx"], errors="coerce")).any():
                errors.append("Some rows have start_idx > end_idx")

    if schema_name == "aggregated_csv":
        if "start_frame" in df.columns and "end_frame" in df.columns:
            if (pd.to_numeric(df["start_frame"], errors="coerce") > pd.to_numeric(df["end_frame"], errors="coerce")).any():
                errors.append("Some rows have start_frame > end_frame")

    # ROM sanity (only if present)
    if "rom" in df.columns:
        rom_vals = pd.to_numeric(df["rom"], errors="coerce")
        if (rom_vals < 0).any():
            errors.append("Some reps have ROM < 0")

    # Duplicate rep_index
    if "rep_index" in df.columns:
        if df["rep_index"].duplicated().any():
            errors.append("Duplicate rep_index values found")

    # Print results
    if errors:
        print("CSV Validation Failed:")
        print(f"  schema_detected: {schema_name}")
        for err in errors:
            print(" -", err)
    else:
        print("CSV Validation Passed.")
        print(f"  schema_detected: {schema_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV to validate")
    args = parser.parse_args()
    validate_csv(args.csv)
