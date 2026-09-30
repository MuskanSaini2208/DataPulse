"""
Clean the DataPulse product dimension from products_raw.csv.

Profiling found 100 unique products, no missing values, no duplicate IDs,
inconsistent category text (case / trailing space), and 12 category vs
sub_category mismatches. This script standardizes labels, corrects a
category only when the same sub_category has a unique majority category
in this file, and writes Data/cleaned/dim_product.csv.
The raw file is never overwritten.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "Data" / "cleaned"
OUTPUT_PATH = OUTPUT_DIR / "dim_product.csv"
QUALITY_PATH = OUTPUT_DIR / "product_quality_report.csv"

RAW_CANDIDATES = [
    PROJECT_ROOT / "Data" / "products_raw.csv",
    PROJECT_ROOT.parent / "products_raw.csv",
]

# Canonical labels already used by the majority of rows in the raw file.
CATEGORY_CANON = {
    "clothing": "Clothing",
    "electronics": "Electronics",
    "furniture": "Furniture",
    "home & kitchen": "Home & Kitchen",
    "office supplies": "Office Supplies",
}

COLUMNS = [
    "product_id",
    "product_name",
    "category",
    "sub_category",
    "cost",
    "selling_price",
]


def resolve_raw_path() -> Path:
    for path in RAW_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("products_raw.csv not found in Data/ or the parent folder.")


def standardize_category(value: str) -> str:
    key = str(value).strip().casefold()
    if key not in CATEGORY_CANON:
        raise ValueError(f"Unexpected category after trim: {value!r}")
    return CATEGORY_CANON[key]


def intended_category_by_subcategory(df: pd.DataFrame) -> dict[str, str | None]:
    """Majority category per sub_category. None if tied or empty."""
    mapping: dict[str, str | None] = {}
    for sub, group in df.groupby("sub_category", sort=True):
        counts = group["category"].value_counts()
        if counts.empty:
            mapping[sub] = None
        elif len(counts) == 1 or counts.iloc[0] > counts.iloc[1]:
            mapping[sub] = counts.index[0]
        else:
            mapping[sub] = None
    return mapping


def main() -> None:
    raw_path = resolve_raw_path()
    df = pd.read_csv(raw_path)
    rows_before = len(df)

    print("=== Product dimension cleaning ===")
    print(f"Source: {raw_path}")
    print(f"Records before cleaning: {rows_before}")

    missing_before = df[COLUMNS].isna().sum()
    print("\nMissing values before cleaning:")
    print(missing_before.to_string())

    dup_counts = df["product_id"].value_counts()
    duplicated_ids = sorted(dup_counts[dup_counts > 1].index.tolist())
    print(f"\nDuplicate product_id values: {duplicated_ids or 'None'}")

    suspicious_price = df[
        (df["cost"] <= 0)
        | (df["selling_price"] <= 0)
        | (df["selling_price"] < df["cost"])
    ]
    print(f"Suspicious prices (non-positive or selling_price < cost): {len(suspicious_price)}")

    quality_rows: list[dict] = []

    # Conflicting duplicate IDs would be flagged; exact copies would be dropped.
    if duplicated_ids:
        exact_ids, conflict_ids = [], []
        for pid, group in df[df["product_id"].isin(duplicated_ids)].groupby("product_id"):
            if group.drop(columns=["product_id"]).nunique().max() <= 1:
                exact_ids.append(pid)
            else:
                conflict_ids.append(pid)
        if conflict_ids:
            raise ValueError(
                f"Conflicting duplicate product_id values: {conflict_ids}. "
                "Not published; resolve before writing dim_product."
            )
        df = df.drop_duplicates(keep="first")
        print(f"Removed exact duplicate rows for: {exact_ids}")

    df = df.copy()
    df["category_raw"] = df["category"]
    df["sub_category_raw"] = df["sub_category"]
    df["category"] = df["category"].map(standardize_category)
    df["sub_category"] = df["sub_category"].astype(str).str.strip()

    format_changed = df["category"] != df["category_raw"].astype(str)
    print("\nCategory labels standardized:")
    print(
        df.loc[format_changed, ["product_id", "category_raw", "category"]]
        .to_string(index=False)
    )

    intended = intended_category_by_subcategory(df)
    print("\nMajority category by sub_category:")
    for sub, cat in intended.items():
        print(f"  {sub}: {cat}")

    resolved, flagged = [], []
    for idx, row in df.iterrows():
        target = intended[row["sub_category"]]
        if target is None:
            flagged.append(row["product_id"])
            quality_rows.append(
                {
                    "product_id": row["product_id"],
                    "issue_type": "category_subcategory_mismatch",
                    "field": "category",
                    "original_value": row["category_raw"],
                    "proposed_value": "",
                    "action": "flagged; subcategory has no unique majority category",
                    "status": "unresolved",
                }
            )
        elif row["category"] != target:
            resolved.append((row["product_id"], row["category"], target, row["sub_category"]))
            quality_rows.append(
                {
                    "product_id": row["product_id"],
                    "issue_type": "category_subcategory_mismatch",
                    "field": "category",
                    "original_value": row["category_raw"],
                    "proposed_value": target,
                    "action": f"set category from sub_category '{row['sub_category']}' majority",
                    "status": "resolved",
                }
            )
            df.at[idx, "category"] = target

    print(f"\nMismatches resolved: {len(resolved)}")
    for pid, old, new, sub in resolved:
        print(f"  {pid}: {old} -> {new} (sub_category={sub})")
    print(f"Mismatches flagged: {flagged or 'None'}")

    for _, row in suspicious_price.iterrows():
        quality_rows.append(
            {
                "product_id": row["product_id"],
                "issue_type": "suspicious_price",
                "field": "selling_price",
                "original_value": f"cost={row['cost']}; selling_price={row['selling_price']}",
                "proposed_value": "",
                "action": "flagged; prices left unchanged",
                "status": "unresolved",
            }
        )

    cleaned = (
        df[COLUMNS]
        .sort_values("product_id")
        .reset_index(drop=True)
    )

    # --- Validation --------------------------------------------------------
    remaining_dupes = cleaned["product_id"].duplicated().sum()
    missing_after = cleaned.isna().sum()
    remaining_mismatch = []
    intended_after = intended_category_by_subcategory(cleaned)
    for _, row in cleaned.iterrows():
        target = intended_after[row["sub_category"]]
        if target is None or row["category"] != target:
            remaining_mismatch.append(row["product_id"])
    extra_space = cleaned["category"].str.strip().ne(cleaned["category"]) | cleaned[
        "sub_category"
    ].str.strip().ne(cleaned["sub_category"])
    unexpected_category = sorted(
        set(cleaned["category"]) - set(CATEGORY_CANON.values())
    )

    print(f"\nRecords after cleaning: {len(cleaned)}")
    print("\n=== Validation ===")
    print(f"Duplicate product_id rows remaining: {remaining_dupes}")
    print("Missing values after cleaning:")
    print(missing_after.to_string())
    print(f"Category/sub_category mismatches remaining: {remaining_mismatch or 'None'}")
    print(f"Leading/trailing spaces remaining: {int(extra_space.sum())}")
    print(f"Unexpected category labels: {unexpected_category or 'None'}")
    print(f"Unique product_id count: {cleaned['product_id'].nunique()}")

    if remaining_dupes != 0:
        raise ValueError("Validation failed: duplicate product_id values remain.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_PATH, index=False)

    quality = pd.DataFrame(quality_rows)
    if quality.empty:
        quality = pd.DataFrame(
            columns=[
                "product_id",
                "issue_type",
                "field",
                "original_value",
                "proposed_value",
                "action",
                "status",
            ]
        )
    quality.to_csv(QUALITY_PATH, index=False)
    print(f"\nWrote {OUTPUT_PATH}")
    print(f"Wrote {QUALITY_PATH}")


if __name__ == "__main__":
    main()
