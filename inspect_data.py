import glob
import pandas as pd

files = sorted([f for f in glob.glob("data/minute/*.parquet") if not f.endswith("_manifest.json")])
print("parquet files:", len(files))

# 随机看 3 个
for f in files[:3]:
    df = pd.read_parquet(f)
    print("\n==", f)
    print("rows:", len(df), "cols:", list(df.columns))
    print(df.head(3))
    print(df.tail(3))

# 看 AAPL 的日期覆盖
aapl = pd.read_parquet("data/minute/AAPL.parquet").sort_values("timestamp")
aapl["date"] = aapl["timestamp"].dt.date
print("\nAAPL dates:", aapl["date"].unique()[:20], "count:", aapl["date"].nunique())
print("AAPL bars/day (last 5):")
print(aapl.groupby("date").size().tail(5))
