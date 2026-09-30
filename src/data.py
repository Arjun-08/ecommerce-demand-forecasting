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
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    print(f"[DATA] Raw shape: {df.shape}")
    print(f"[DATA] Available columns: {list(df.columns)}")

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    )

    df["UnitPrice"] = pd.to_numeric(
        df["UnitPrice"],
        errors="coerce"
    )

    before = len(df)

    # Keep only valid fulfilled sales.
    df = df[
        df["InvoiceDate"].notna()
        & df["Description"].notna()
        & df["Quantity"].notna()
        & df["UnitPrice"].notna()
        & (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
    ].copy()

    df["Description"] = (
        df["Description"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    after = len(df)

    print(f"[DATA] Valid demand rows: {after:,}")
    print(f"[DATA] Removed rows: {before - after:,}")
    print(
        f"[DATA] Date range: "
        f"{df['InvoiceDate'].min()} -> {df['InvoiceDate'].max()}"
    )

    print(
        f"[DATA] Unique products: "
        f"{df['Description'].nunique():,}"
    )

    return df


def choose_product(
    df: pd.DataFrame,
    min_active_days: int = 90,
    min_total_units: int = 500,
) -> tuple[str, str]:

    print(
        "[DATA] Selecting a product with enough "
        "history for forecasting..."
    )

    daily = (
        df.assign(
            date=df["InvoiceDate"].dt.floor("D")
        )
        .groupby(
            ["Description", "date"],
            as_index=False
        )["Quantity"]
        .sum()
    )

    stats = (
        daily
        .groupby("Description")
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
            "Lower min_active_days or min_total_units."
        )

    # Prefer products with long histories and substantial demand.
    candidates = candidates.sort_values(
        ["active_days", "total_units"],
        ascending=False
    )

    row = candidates.iloc[0]

    description = str(row["Description"])

    print(f"[DATA] Selected product: {description}")
    print(
        f"[DATA] Active days: "
        f"{int(row['active_days']):,}"
    )
    print(
        f"[DATA] Total units: "
        f"{row['total_units']:,.0f}"
    )

    # Since StockCode is unavailable in the downloaded table,
    # use the product description as the product identifier.
    return description, description


def build_daily_series(
    df: pd.DataFrame,
    product_id: str,
) -> pd.DataFrame:

    product = df[
        df["Description"].astype(str) == str(product_id)
    ].copy()

    if product.empty:
        raise ValueError(
            f"Product {product_id} not found."
        )

    daily = (
        product.assign(
            date=product["InvoiceDate"].dt.floor("D")
        )
        .groupby("date")["Quantity"]
        .sum()
        .sort_index()
        .rename("sales")
        .to_frame()
    )

    # Explicitly represent days with no sales as zero demand.
    full_index = pd.date_range(
        daily.index.min(),
        daily.index.max(),
        freq="D",
    )

    daily = daily.reindex(
        full_index,
        fill_value=0.0
    )

    daily.index.name = "date"

    print(
        f"[DATA] Product time series length: "
        f"{len(daily):,} days"
    )

    print(
        f"[DATA] Zero-demand days: "
        f"{(daily['sales'] == 0).sum():,}"
    )

    print(
        f"[DATA] Mean daily demand: "
        f"{daily['sales'].mean():.2f}"
    )

    return daily