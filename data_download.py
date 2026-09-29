from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = RAW_DIR / "online_retail.csv"

print("=" * 72)
print("DOWNLOADING UCI ONLINE RETAIL DATASET")
print("=" * 72)

if OUTPUT.exists():
    print(f"[INFO] Dataset already exists: {OUTPUT}")
    print("[INFO] Delete it manually if you want to download a fresh copy.")
    raise SystemExit(0)

print("[1/3] Fetching dataset from UCI Machine Learning Repository...")
dataset = fetch_ucirepo(id=352)

print("[2/3] Reading dataset...")
df = dataset.data.features.copy()

print(f"[INFO] Rows: {len(df):,}")
print(f"[INFO] Columns: {list(df.columns)}")

print("[3/3] Saving local working copy...")
df.to_csv(OUTPUT, index=False)

print(f"[DONE] Saved to: {OUTPUT}")
print()
print("IMPORTANT:")
print("- The raw dataset is intentionally gitignored.")
print("- Keep the UCI attribution in the project README.")
print("- Do not commit or redistribute the raw dataset with this repository.")
