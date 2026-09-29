from pathlib import Path
import pandas as pd


def load_transactions(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    print(f"[DATA] Loading transactions from {path}...")

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python download_data.py` first."
        )

    df = pd.read_csv(path, low_memory=False)

    required = {
        "InvoiceNo", "StockCode", "Description",
        "Quantity", "InvoiceDate", "UnitPrice"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    print(f"[DATA] Raw shape: {df.shape}")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    before = len(df)

    # Demand is defined here as fulfilled positive-quantity sales.
    # Cancellation invoices and returns are excluded rather than treated as demand.
    invoice_str = df["InvoiceNo"].astype(str)
    df = df[
        df["InvoiceDate"].notna()
        & df["StockCode"].notna()
        & df["Quantity"].notna()
        & df["UnitPrice"].notna()
        & (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & (~invoice_str.str.upper().str.startswith("C"))
    ].copy()

    # Remove obvious non-product/service rows where the stock code is not useful
    # for product-level demand forecasting.
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Description"] = df["Description"].fillna("Unknown product").astype(str).str.strip()

    after = len(df)

    print(f"[DATA] Valid demand rows: {after:,}")
    print(f"[DATA] Removed rows: {before - after:,}")
    print(f"[DATA] Date range: {df['InvoiceDate'].min()} -> {df['InvoiceDate'].max()}")
    print(f"[DATA] Unique products: {df['StockCode'].nunique():,}")

    return df


def choose_product(
    df: pd.DataFrame,
    min_active_days: int = 90,
    min_total_units: int = 500,
) -> tuple[str, str]:
    print("[DATA] Selecting a product with enough history for forecasting...")

    daily = (
        df.assign(date=df["InvoiceDate"].dt.floor("D"))
        .groupby(["StockCode", "Description", "date"], as_index=False)["Quantity"]
        .sum()
    )

    stats = (
        daily.groupby(["StockCode", "Description"])
        .agg(
            active_days=("date", "nunique"),
            total_units=("Quantity", "sum"),
        )
        .reset_index()
    )

    candidates = stats[
        (stats["active_days"] >= min_active_days)
        & (stats["total_units"] >= min_total_units)
    ].copy()

    if candidates.empty:
        raise RuntimeError(
            "No product satisfies the selection criteria. "
            "Lower min_active_days/min_total_units."
        )

    candidates = candidates.sort_values(
        ["active_days", "total_units"], ascending=False
    )

    row = candidates.iloc[0]
    product_id = str(row["StockCode"])
    description = str(row["Description"])

    print(f"[DATA] Selected product: {product_id}")
    print(f"[DATA] Description: {description}")
    print(f"[DATA] Active days: {int(row['active_days']):,}")
    print(f"[DATA] Total units: {row['total_units']:,.0f}")

    return product_id, description


def build_daily_series(
    df: pd.DataFrame,
    product_id: str,
) -> pd.DataFrame:
    product = df[df["StockCode"].astype(str) == str(product_id)].copy()

    if product.empty:
        raise ValueError(f"Product {product_id} not found.")

    daily = (
        product.assign(date=product["InvoiceDate"].dt.floor("D"))
        .groupby("date", as_index=True)["Quantity"]
        .sum()
        .sort_index()
        .rename("sales")
        .to_frame()
    )

    # Fill dates with zero demand. This is important because a missing date in
    # transaction data can mean "no sale", not "unknown".
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index, fill_value=0.0)
    daily.index.name = "date"

    print(f"[DATA] Product time series length: {len(daily):,} days")
    print(f"[DATA] Zero-demand days: {(daily['sales'] == 0).sum():,}")
    print(f"[DATA] Mean daily demand: {daily['sales'].mean():.2f}")

    return daily
