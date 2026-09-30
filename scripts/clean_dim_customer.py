"""
Clean the DataPulse customer dimension from customers_raw.csv.

Profiling (prior step) found:
- 2,010 rows, 6 columns, no missing values
- 2,000 unique customer_id values
- 10 customer_id values appearing twice as exact full-row copies

This script verifies those findings, removes exact duplicate rows only,
and writes one row per customer_id to Data/cleaned/dim_customer.csv.
The raw file is never overwritten.
"""

from pathlib import Path

import pandas as pd

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Raw extract lives beside the project folder (unchanged source file).
RAW_CUSTOMERS = (
    PROJECT_ROOT.parent / "customers_raw.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "Data" / "cleaned"
OUTPUT_PATH = OUTPUT_DIR / "dim_customer.csv"

# Duplicate IDs reported during profiling — re-verified below, not assumed.
PROFILED_DUP_IDS = [
    "C00205",
    "C00285",
    "C00695",
    "C01049",
    "C01073",
    "C01725",
    "C01759",
    "C01867",
    "C01887",
    "C01927",
]

CUSTOMER_COLUMNS = [
    "customer_id",
    "customer_name",
    "gender",
    "age",
    "customer_segment",
    "signup_date",
]


def classify_duplicate_ids(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split duplicate customer_id values into exact copies vs conflicts."""
    dup_mask = df.duplicated(subset=["customer_id"], keep=False)
    dup_rows = df.loc[dup_mask].copy()

    exact_ids: list[str] = []
    conflicting_ids: list[str] = []

    for customer_id, group in dup_rows.groupby("customer_id", sort=True):
        # Exact copy: every attribute is identical across the group.
        if group.drop(columns=["customer_id"]).nunique().max() <= 1:
            exact_ids.append(customer_id)
        else:
            conflicting_ids.append(customer_id)

    return exact_ids, conflicting_ids


def main() -> None:
    if not RAW_CUSTOMERS.exists():
        raise FileNotFoundError(f"Raw customer file not found: {RAW_CUSTOMERS}")

    df = pd.read_csv(RAW_CUSTOMERS, dtype=str)
    rows_before = len(df)

    print("=== Customer dimension cleaning ===")
    print(f"Source: {RAW_CUSTOMERS}")
    print(f"Records before cleaning: {rows_before}")
    print(f"Columns: {list(df.columns)}")

    missing_before = df[CUSTOMER_COLUMNS].isna() | (
        df[CUSTOMER_COLUMNS].apply(lambda col: col.str.strip().eq(""))
    )
    missing_counts = missing_before.sum()
    print("\nMissing / blank values before cleaning:")
    print(missing_counts.to_string())

    # --- Duplicate ID investigation ----------------------------------------
    dup_id_counts = df["customer_id"].value_counts()
    duplicated_ids = sorted(dup_id_counts[dup_id_counts > 1].index.tolist())
    print(f"\nDuplicate customer_id values found: {len(duplicated_ids)}")
    print(duplicated_ids)

    if duplicated_ids != PROFILED_DUP_IDS:
        print(
            "WARNING: duplicate IDs differ from the profiling list. "
            "Using IDs actually present in the file."
        )

    exact_ids, conflicting_ids = classify_duplicate_ids(df)

    print("\nExact-copy duplicate IDs (safe to collapse to one row):")
    print(exact_ids)
    print("Conflicting duplicate IDs (flag for review; do not drop arbitrarily):")
    print(conflicting_ids if conflicting_ids else "None")

    if conflicting_ids:
        print("\n--- FLAG FOR REVIEW: conflicting attributes for the same customer_id ---")
        conflict_rows = df[df["customer_id"].isin(conflicting_ids)].sort_values(
            ["customer_id"]
        )
        print(conflict_rows.to_string(index=False))
        raise ValueError(
            "Conflicting duplicate customer_id values found. "
            "No row was deleted. Resolve these IDs before publishing dim_customer."
        )

    # Decision: drop exact duplicate rows only. No attributes are imputed.
    cleaned = df.drop_duplicates(keep="first")

    cleaned = cleaned[CUSTOMER_COLUMNS].sort_values("customer_id").reset_index(drop=True)

    rows_after = len(cleaned)
    print(f"\nRecords after cleaning: {rows_after}")
    print(f"Rows removed (exact duplicates): {rows_before - rows_after}")

    # --- Validation --------------------------------------------------------
    remaining_dupes = cleaned["customer_id"].duplicated().sum()
    missing_after = cleaned.isna() | cleaned.apply(lambda col: col.str.strip().eq(""))
    missing_after_counts = missing_after.sum()

    print("\n=== Validation ===")
    print(f"Duplicate customer_id rows remaining: {remaining_dupes}")
    print("Missing / blank values after cleaning:")
    print(missing_after_counts.to_string())
    print(f"Unique customer_id count: {cleaned['customer_id'].nunique()}")

    if remaining_dupes != 0:
        raise ValueError(
            "Validation failed: cleaned customer dimension still has duplicate IDs. "
            "Conflicting duplicates were retained for review."
        )

    unexpected_missing = missing_after_counts[missing_after_counts > 0]
    if len(unexpected_missing) > 0:
        print(
            "WARNING: missing values remain. None were filled; "
            "no reliable business rule exists to invent customer attributes."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote cleaned customer dimension: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
