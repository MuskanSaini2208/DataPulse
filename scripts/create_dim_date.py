"""Create the DataPulse date dimension from valid cleaned order dates only.

The output has one row per date present in fact_orders.csv. Dates flagged as
invalid, blank dates, and any unexpected validity flag values cause no inferred
calendar rows: invalid source dates are excluded and malformed valid dates stop
the process for review. Quarter is stored as the readable label Q1 through Q4.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FACT_ORDERS_PATH = PROJECT_ROOT / "Data" / "cleaned" / "fact_orders.csv"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "cleaned"
OUTPUT_PATH = OUTPUT_DIR / "dim_date.csv"

REQUIRED_FACT_COLUMNS = ["order_date", "is_order_date_valid"]
DATE_DIMENSION_COLUMNS = [
    "date_key",
    "date",
    "day",
    "month",
    "month_name",
    "quarter",
    "year",
]


def require_columns(frame: pd.DataFrame, expected: list[str]) -> None:
    """Confirm the Stage 4 fact table has the fields needed for this dimension."""
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError(f"fact_orders.csv is missing required columns: {missing}")


def build_date_dimension(fact_orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return one sorted date-dimension row per valid order date."""
    validity_flag = fact_orders["is_order_date_valid"].str.strip().str.casefold()
    unexpected_flags = sorted(set(validity_flag) - {"true", "false"})
    if unexpected_flags:
        raise ValueError(
            "Unexpected is_order_date_valid values found: "
            f"{unexpected_flags}. Review the fact table before creating dim_date."
        )

    valid_orders = fact_orders.loc[validity_flag.eq("true"), ["order_date"]].copy()
    if valid_orders.empty:
        raise ValueError("No valid order dates are available to build dim_date.")
    if valid_orders["order_date"].str.strip().eq("").any():
        raise ValueError("A date marked valid is blank; no date dimension was written.")

    parsed_dates = pd.to_datetime(
        valid_orders["order_date"], format="%Y-%m-%d", errors="coerce"
    )
    if parsed_dates.isna().any():
        bad_values = valid_orders.loc[parsed_dates.isna(), "order_date"].unique().tolist()
        raise ValueError(
            "A date marked valid cannot be parsed as YYYY-MM-DD: "
            f"{bad_values[:10]}"
        )

    unique_dates = pd.Series(parsed_dates.drop_duplicates().sort_values().reset_index(drop=True))
    dimension = pd.DataFrame({"date": unique_dates.dt.strftime("%Y-%m-%d")})
    dimension["date_key"] = unique_dates.dt.strftime("%Y%m%d").astype("int64")
    dimension["day"] = unique_dates.dt.day.astype("int64")
    dimension["month"] = unique_dates.dt.month.astype("int64")
    dimension["month_name"] = unique_dates.dt.month_name()
    dimension["quarter"] = "Q" + unique_dates.dt.quarter.astype(str)
    dimension["year"] = unique_dates.dt.year.astype("int64")
    dimension = dimension[DATE_DIMENSION_COLUMNS]

    return dimension, valid_orders["order_date"]


def validate_date_dimension(
    dimension: pd.DataFrame, valid_order_dates: pd.Series
) -> None:
    """Validate uniqueness, completeness, and order-date coverage."""
    if dimension.empty:
        raise ValueError("dim_date is empty.")
    if dimension.isna().any().any() or dimension.eq("").any().any():
        raise ValueError("dim_date contains missing or blank values.")
    if dimension["date_key"].duplicated().any():
        raise ValueError("dim_date contains duplicate date_key values.")
    if dimension["date"].duplicated().any():
        raise ValueError("dim_date contains duplicate date values.")

    dimension_dates = set(dimension["date"])
    order_dates = set(valid_order_dates)
    missing_dates = sorted(order_dates - dimension_dates)
    unexpected_dates = sorted(dimension_dates - order_dates)
    if missing_dates or unexpected_dates:
        raise ValueError(
            "Date coverage validation failed. "
            f"Missing: {missing_dates[:10]}; unexpected: {unexpected_dates[:10]}"
        )


def main() -> None:
    if not FACT_ORDERS_PATH.exists():
        raise FileNotFoundError(f"Cleaned order fact table not found: {FACT_ORDERS_PATH}")

    fact_orders = pd.read_csv(FACT_ORDERS_PATH, dtype=str, keep_default_na=False)
    require_columns(fact_orders, REQUIRED_FACT_COLUMNS)

    dimension, valid_order_dates = build_date_dimension(fact_orders)
    validate_date_dimension(dimension, valid_order_dates)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dimension.to_csv(OUTPUT_PATH, index=False)

    invalid_date_rows = int(
        fact_orders["is_order_date_valid"].str.strip().str.casefold().eq("false").sum()
    )
    print("=== Date dimension creation complete ===")
    print(f"Fact rows read: {len(fact_orders)}")
    print(f"Valid order-date rows used: {len(valid_order_dates)}")
    print(f"Invalid-date rows excluded: {invalid_date_rows}")
    print(f"Unique dates written: {len(dimension)}")
    print("Duplicate date_key values: 0")
    print("Duplicate date values: 0")
    print("Missing values: 0")
    print("Valid order-date coverage: complete")
    print(f"Wrote date dimension: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
