"""Clean the DataPulse orders extract into a reviewable fact table.

This Stage 4 transformation removes only exact full-row duplicate
transactions. Invalid dates and incomplete dimension keys are not invented or
imputed: the fact table retains the record and supplies flags for review.
Negative quantities are labeled with ``is_return`` as an observed condition,
not a confirmed business classification.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ORDERS_PATH = PROJECT_ROOT / "Data" / "raw" / "orders_raw.csv"
CUSTOMER_DIM_PATH = PROJECT_ROOT / "Data" / "cleaned" / "dim_customer.csv"
PRODUCT_DIM_PATH = PROJECT_ROOT / "Data" / "cleaned" / "dim_product.csv"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "cleaned"
FACT_OUTPUT_PATH = OUTPUT_DIR / "fact_orders.csv"
AUDIT_OUTPUT_PATH = OUTPUT_DIR / "order_quality_report.csv"

REQUIRED_ORDER_COLUMNS = [
    "order_id",
    "order_date",
    "customer_id",
    "product_id",
    "region_id",
    "quantity",
    "discount",
    "sales_amount",
    "cost_amount",
    "profit",
]


def require_columns(frame: pd.DataFrame, expected: list[str], source: Path) -> None:
    """Fail clearly if a source schema no longer supports this transformation."""
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def conflicting_duplicate_order_ids(frame: pd.DataFrame) -> list[str]:
    """Return duplicate order IDs whose rows are not identical copies."""
    duplicate_rows = frame[frame.duplicated(subset=["order_id"], keep=False)]
    conflicts: list[str] = []

    for order_id, group in duplicate_rows.groupby("order_id", sort=True):
        if len(group.drop_duplicates()) > 1:
            conflicts.append(order_id)

    return conflicts


def make_audit_report(
    raw_rows: int,
    exact_duplicates_removed: int,
    fact: pd.DataFrame,
    missing_customer_count: int,
    invalid_customer_count: int,
    missing_product_count: int,
    invalid_product_count: int,
) -> pd.DataFrame:
    """Create a compact, CSV-friendly audit trail for the Stage 4 decisions."""
    report_rows = [
        {
            "section": "lineage",
            "metric": "source_file",
            "value": str(RAW_ORDERS_PATH.relative_to(PROJECT_ROOT)),
            "details": "Raw source retained unchanged.",
        },
        {
            "section": "row_counts",
            "metric": "raw_rows",
            "value": raw_rows,
            "details": "Rows read from the raw orders extract.",
        },
        {
            "section": "row_counts",
            "metric": "exact_duplicates_removed",
            "value": exact_duplicates_removed,
            "details": "Removed only when every source column was identical.",
        },
        {
            "section": "row_counts",
            "metric": "cleaned_fact_rows",
            "value": len(fact),
            "details": "Unique transactions retained in the cleaned fact table.",
        },
        {
            "section": "validation",
            "metric": "duplicate_order_ids_remaining",
            "value": int(fact["order_id"].duplicated().sum()),
            "details": "The current source grain is one row per order_id after exact de-duplication.",
        },
        {
            "section": "dates",
            "metric": "invalid_order_dates",
            "value": int(fact["is_order_date_valid"].eq(False).sum()),
            "details": "Invalid values are blank in order_date; the original text remains in order_date_raw.",
        },
        {
            "section": "customer_keys",
            "metric": "missing_customer_ids",
            "value": missing_customer_count,
            "details": "Blank IDs retained and flagged; no unknown customer was assigned.",
        },
        {
            "section": "customer_keys",
            "metric": "nonblank_customer_ids_not_in_dim_customer",
            "value": invalid_customer_count,
            "details": "Checked against Data/cleaned/dim_customer.csv.",
        },
        {
            "section": "product_keys",
            "metric": "missing_product_ids",
            "value": missing_product_count,
            "details": "Blank IDs retained and flagged; no unknown product was assigned.",
        },
        {
            "section": "product_keys",
            "metric": "nonblank_product_ids_not_in_dim_product",
            "value": invalid_product_count,
            "details": "Checked against Data/cleaned/dim_product.csv.",
        },
        {
            "section": "quantities",
            "metric": "negative_quantity_rows",
            "value": int(fact["is_return"].sum()),
            "details": "is_return is an observation based on quantity < 0, not confirmation of a return.",
        },
        {
            "section": "decision",
            "metric": "duplicate_handling",
            "value": "exact_full_row_only",
            "details": "Conflicting duplicate order_id values stop the script instead of being removed.",
        },
        {
            "section": "decision",
            "metric": "invalid_date_handling",
            "value": "preserved_with_flag",
            "details": "No replacement date was inferred or created.",
        },
        {
            "section": "decision",
            "metric": "foreign_key_handling",
            "value": "preserved_with_flags",
            "details": "Missing or unmatched keys remain visible for later business review.",
        },
    ]
    return pd.DataFrame(report_rows)


def main() -> None:
    for path in (RAW_ORDERS_PATH, CUSTOMER_DIM_PATH, PRODUCT_DIM_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    raw_orders = pd.read_csv(RAW_ORDERS_PATH, dtype=str, keep_default_na=False)
    customers = pd.read_csv(CUSTOMER_DIM_PATH, dtype=str, keep_default_na=False)
    products = pd.read_csv(PRODUCT_DIM_PATH, dtype=str, keep_default_na=False)

    require_columns(raw_orders, REQUIRED_ORDER_COLUMNS, RAW_ORDERS_PATH)
    require_columns(customers, ["customer_id"], CUSTOMER_DIM_PATH)
    require_columns(products, ["product_id"], PRODUCT_DIM_PATH)

    raw_rows = len(raw_orders)
    conflicts = conflicting_duplicate_order_ids(raw_orders)
    if conflicts:
        raise ValueError(
            "Conflicting rows share an order_id; no transactions were removed. "
            f"Review these IDs: {conflicts[:10]}"
        )

    fact = raw_orders.drop_duplicates(keep="first").copy()
    exact_duplicates_removed = raw_rows - len(fact)

    if fact["order_id"].str.strip().eq("").any():
        raise ValueError("Blank order_id values found; a fact-table primary key cannot be established.")
    if fact["order_id"].duplicated().any():
        raise ValueError("Duplicate order_id values remain after exact de-duplication.")

    # Normalize keys only for whitespace. The raw source remains untouched.
    for column in ("order_id", "customer_id", "product_id", "region_id"):
        fact[column] = fact[column].str.strip()

    fact["order_date_raw"] = fact["order_date"]
    parsed_dates = pd.to_datetime(
        fact["order_date_raw"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    fact["is_order_date_valid"] = parsed_dates.notna()
    fact["order_date"] = parsed_dates.dt.strftime("%Y-%m-%d").fillna("")

    fact["quantity"] = pd.to_numeric(fact["quantity"], errors="raise").astype("int64")
    for column in ("discount", "sales_amount", "cost_amount", "profit"):
        fact[column] = pd.to_numeric(fact[column], errors="raise")

    customer_ids = set(customers["customer_id"].str.strip())
    product_ids = set(products["product_id"].str.strip())

    fact["customer_id_missing"] = fact["customer_id"].eq("")
    fact["customer_id_in_dim"] = (
        ~fact["customer_id_missing"] & fact["customer_id"].isin(customer_ids)
    )
    fact["product_id_missing"] = fact["product_id"].eq("")
    fact["product_id_in_dim"] = (
        ~fact["product_id_missing"] & fact["product_id"].isin(product_ids)
    )
    fact["is_return"] = fact["quantity"].lt(0)

    review_required = (
        ~fact["is_order_date_valid"]
        | fact["customer_id_missing"]
        | ~fact["customer_id_in_dim"]
        | fact["product_id_missing"]
        | ~fact["product_id_in_dim"]
        | fact["is_return"]
    )
    fact["requires_review"] = review_required

    missing_customer_count = int(fact["customer_id_missing"].sum())
    invalid_customer_count = int(
        (~fact["customer_id_missing"] & ~fact["customer_id_in_dim"]).sum()
    )
    missing_product_count = int(fact["product_id_missing"].sum())
    invalid_product_count = int(
        (~fact["product_id_missing"] & ~fact["product_id_in_dim"]).sum()
    )

    audit = make_audit_report(
        raw_rows=raw_rows,
        exact_duplicates_removed=exact_duplicates_removed,
        fact=fact,
        missing_customer_count=missing_customer_count,
        invalid_customer_count=invalid_customer_count,
        missing_product_count=missing_product_count,
        invalid_product_count=invalid_product_count,
    )

    if len(fact) != raw_rows - exact_duplicates_removed:
        raise ValueError("Row-count reconciliation failed.")
    if fact.duplicated(subset=REQUIRED_ORDER_COLUMNS).any():
        raise ValueError("Exact duplicate transactions remain after cleaning.")
    if not fact["is_return"].eq(fact["quantity"].lt(0)).all():
        raise ValueError("Return flag validation failed.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fact.to_csv(FACT_OUTPUT_PATH, index=False)
    audit.to_csv(AUDIT_OUTPUT_PATH, index=False)

    print("=== Order fact cleaning complete ===")
    print(f"Raw rows: {raw_rows}")
    print(f"Exact duplicates removed: {exact_duplicates_removed}")
    print(f"Cleaned fact rows: {len(fact)}")
    print(f"Missing customer IDs: {missing_customer_count}")
    print(f"Unmatched nonblank customer IDs: {invalid_customer_count}")
    print(f"Missing product IDs: {missing_product_count}")
    print(f"Unmatched nonblank product IDs: {invalid_product_count}")
    print(f"Invalid order dates: {int((~fact['is_order_date_valid']).sum())}")
    print(f"Negative quantities / is_return=True: {int(fact['is_return'].sum())}")
    print(f"Wrote fact table: {FACT_OUTPUT_PATH}")
    print(f"Wrote audit report: {AUDIT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
