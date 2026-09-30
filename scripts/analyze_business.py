"""Produce Stage 6 business-analysis outputs for DataPulse.

All results use valid order dates only. Product-level outputs require a
matching product dimension record; regional outputs require a matching region
record. Negative-quantity transactions remain in revenue and profit totals as
net values. Their is_return flag is an observed condition, not confirmation
that the transaction is a business return.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_DIR = PROJECT_ROOT / "Data" / "cleaned"
ANALYSIS_DIR = PROJECT_ROOT / "Data" / "analysis"

FACT_PATH = CLEANED_DIR / "fact_orders.csv"
PRODUCT_PATH = CLEANED_DIR / "dim_product.csv"
DATE_PATH = CLEANED_DIR / "dim_date.csv"
REGION_PATH = PROJECT_ROOT / "Data" / "raw" / "regions_raw.csv"

TOP_PRODUCTS_PATH = ANALYSIS_DIR / "top_products_by_revenue.csv"
REGIONAL_PROFIT_PATH = ANALYSIS_DIR / "regional_profit.csv"
MONTHLY_TRENDS_PATH = ANALYSIS_DIR / "monthly_revenue_profit.csv"
LOW_MARGIN_PRODUCTS_PATH = ANALYSIS_DIR / "high_revenue_low_margin_products.csv"

TOP_PRODUCT_COUNT = 10
HIGH_REVENUE_QUANTILE = 0.75
LOW_MARGIN_QUANTILE = 0.25


def require_columns(frame: pd.DataFrame, expected: list[str], source: Path) -> None:
    """Confirm the source schema supports the requested analysis."""
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def aggregate_metrics(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Aggregate net financial measures and calculate margin from totals."""
    result = (
        frame.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            net_revenue=("sales_amount", "sum"),
            net_profit=("profit", "sum"),
            transaction_count=("order_id", "size"),
            negative_quantity_rows=("is_return", "sum"),
        )
        .copy()
    )
    result["profit_margin_pct"] = (
        result["net_profit"].div(result["net_revenue"]).mul(100)
    )
    return result


def round_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Round displayed financial values without changing selection logic."""
    result = frame.copy()
    for column in ("net_revenue", "net_profit", "profit_margin_pct"):
        if column in result:
            result[column] = result[column].round(2)
    return result


def validate_outputs(
    top_products: pd.DataFrame,
    regional_profit: pd.DataFrame,
    monthly_trends: pd.DataFrame,
    low_margin_products: pd.DataFrame,
    product_fact: pd.DataFrame,
    region_fact: pd.DataFrame,
    dated_fact: pd.DataFrame,
    revenue_threshold: float,
    margin_threshold: float,
) -> None:
    """Check ranking, uniqueness, coverage, and threshold adherence."""
    expected_top_count = min(TOP_PRODUCT_COUNT, product_fact["product_id"].nunique())
    if len(top_products) != expected_top_count or top_products["product_id"].duplicated().any():
        raise ValueError("Top-products validation failed.")
    if not top_products["net_revenue"].is_monotonic_decreasing:
        raise ValueError("Top-products output is not sorted by revenue.")

    if regional_profit["region"].duplicated().any():
        raise ValueError("Regional-profit output has duplicate region rows.")
    if not regional_profit["net_profit"].is_monotonic_decreasing:
        raise ValueError("Regional-profit output is not sorted by profit.")
    if round(regional_profit["net_profit"].sum(), 2) != round(region_fact["profit"].sum(), 2):
        raise ValueError("Regional-profit total does not reconcile to the fact table.")

    if monthly_trends.duplicated(subset=["year", "month"]).any():
        raise ValueError("Monthly-trends output has duplicate year/month rows.")
    if round(monthly_trends["net_revenue"].sum(), 2) != round(dated_fact["sales_amount"].sum(), 2):
        raise ValueError("Monthly revenue does not reconcile to valid-dated facts.")
    if round(monthly_trends["net_profit"].sum(), 2) != round(dated_fact["profit"].sum(), 2):
        raise ValueError("Monthly profit does not reconcile to valid-dated facts.")

    if low_margin_products["product_id"].duplicated().any():
        raise ValueError("Low-margin output has duplicate product rows.")
    if not low_margin_products.empty:
        if (low_margin_products["net_revenue"] < revenue_threshold).any():
            raise ValueError("A low-margin product falls below the revenue threshold.")
        if (low_margin_products["profit_margin_pct"] > margin_threshold).any():
            raise ValueError("A low-margin product exceeds the margin threshold.")


def main() -> None:
    for path in (FACT_PATH, PRODUCT_PATH, DATE_PATH, REGION_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required analysis input not found: {path}")

    fact = pd.read_csv(FACT_PATH, dtype=str, keep_default_na=False)
    products = pd.read_csv(PRODUCT_PATH, dtype=str, keep_default_na=False)
    dates = pd.read_csv(DATE_PATH, dtype=str, keep_default_na=False)
    regions = pd.read_csv(REGION_PATH, dtype=str, keep_default_na=False)

    require_columns(
        fact,
        [
            "order_id",
            "order_date",
            "product_id",
            "region_id",
            "sales_amount",
            "profit",
            "is_order_date_valid",
            "is_return",
        ],
        FACT_PATH,
    )
    require_columns(products, ["product_id", "product_name", "category", "sub_category"], PRODUCT_PATH)
    require_columns(dates, ["date", "year", "month", "month_name"], DATE_PATH)
    require_columns(regions, ["region_id", "region"], REGION_PATH)

    if products["product_id"].duplicated().any():
        raise ValueError("dim_product.csv has duplicate product_id values.")
    if dates["date"].duplicated().any():
        raise ValueError("dim_date.csv has duplicate date values.")
    if regions["region_id"].duplicated().any():
        raise ValueError("regions_raw.csv has duplicate region_id values.")

    for column in ("sales_amount", "profit"):
        fact[column] = pd.to_numeric(fact[column], errors="raise")
    fact["is_return"] = fact["is_return"].str.strip().str.casefold().eq("true")

    dated_fact = fact.loc[fact["is_order_date_valid"].str.strip().str.casefold().eq("true")].copy()
    if dated_fact["order_date"].eq("").any():
        raise ValueError("A valid-dated fact row has a blank order_date.")

    product_fact = dated_fact.merge(
        products[["product_id", "product_name", "category", "sub_category"]],
        on="product_id",
        how="inner",
        validate="many_to_one",
    )
    region_fact = dated_fact.merge(
        regions[["region_id", "region"]],
        on="region_id",
        how="inner",
        validate="many_to_one",
    )
    dated_fact = dated_fact.merge(
        dates[["date", "year", "month", "month_name"]],
        left_on="order_date",
        right_on="date",
        how="inner",
        validate="many_to_one",
    ).drop(columns=["date"])

    top_products = aggregate_metrics(
        product_fact, ["product_id", "product_name", "category", "sub_category"]
    )
    top_products = top_products.sort_values(
        ["net_revenue", "product_id"], ascending=[False, True]
    ).head(TOP_PRODUCT_COUNT)
    top_products.insert(0, "revenue_rank", range(1, len(top_products) + 1))
    top_products = round_metrics(top_products).reset_index(drop=True)

    regional_profit = aggregate_metrics(region_fact, ["region"])
    regional_profit = regional_profit.sort_values(
        ["net_profit", "region"], ascending=[False, True]
    )
    regional_profit.insert(0, "profit_rank", range(1, len(regional_profit) + 1))
    regional_profit = round_metrics(regional_profit).reset_index(drop=True)

    monthly_trends = aggregate_metrics(dated_fact, ["year", "month", "month_name"])
    monthly_trends["year"] = pd.to_numeric(monthly_trends["year"], errors="raise")
    monthly_trends["month"] = pd.to_numeric(monthly_trends["month"], errors="raise")
    monthly_trends = round_metrics(
        monthly_trends.sort_values(["year", "month"]).reset_index(drop=True)
    )

    all_product_metrics = aggregate_metrics(
        product_fact, ["product_id", "product_name", "category", "sub_category"]
    )
    margin_eligible = all_product_metrics.loc[all_product_metrics["net_revenue"] > 0].copy()
    revenue_threshold = margin_eligible["net_revenue"].quantile(HIGH_REVENUE_QUANTILE)
    margin_threshold = margin_eligible["profit_margin_pct"].quantile(LOW_MARGIN_QUANTILE)
    low_margin_products = margin_eligible.loc[
        (margin_eligible["net_revenue"] >= revenue_threshold)
        & (margin_eligible["profit_margin_pct"] <= margin_threshold)
    ].copy()
    low_margin_products["revenue_top_25pct_threshold"] = revenue_threshold
    low_margin_products["margin_bottom_25pct_threshold"] = margin_threshold
    low_margin_products = low_margin_products.sort_values(
        ["net_revenue", "profit_margin_pct", "product_id"],
        ascending=[False, True, True],
    )
    low_margin_products = round_metrics(low_margin_products).reset_index(drop=True)

    validate_outputs(
        top_products=top_products,
        regional_profit=regional_profit,
        monthly_trends=monthly_trends,
        low_margin_products=low_margin_products,
        product_fact=product_fact,
        region_fact=region_fact,
        dated_fact=dated_fact,
        revenue_threshold=revenue_threshold,
        margin_threshold=margin_threshold,
    )

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    top_products.to_csv(TOP_PRODUCTS_PATH, index=False)
    regional_profit.to_csv(REGIONAL_PROFIT_PATH, index=False)
    monthly_trends.to_csv(MONTHLY_TRENDS_PATH, index=False)
    low_margin_products.to_csv(LOW_MARGIN_PRODUCTS_PATH, index=False)

    print("=== Stage 6 business analysis complete ===")
    print(f"Valid-date fact rows used for monthly trends: {len(dated_fact)}")
    print(f"Product-matched fact rows used for product outputs: {len(product_fact)}")
    print(f"Region-matched fact rows used for regional output: {len(region_fact)}")
    print(f"Top-product rows written: {len(top_products)}")
    print(f"Regional rows written: {len(regional_profit)}")
    print(f"Monthly rows written: {len(monthly_trends)}")
    print(f"High-revenue, low-margin products: {len(low_margin_products)}")
    print(f"Revenue top-25% threshold: {revenue_threshold:.2f}")
    print(f"Margin bottom-25% threshold: {margin_threshold:.2f}%")


if __name__ == "__main__":
    main()
