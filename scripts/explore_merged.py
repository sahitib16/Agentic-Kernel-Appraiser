# scripts/explore_merged.py
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

KEY_METRICS = [
    "gpu__time_duration.sum",
    "dram__bytes.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
]

CATEGORICALS = ["operation","conv_role","layout","dtype","math_mode","tile_shape_bucket"]

PROBLEM_DIMS = ["M","N","K","H","W","C","R","S"]
TILE_DIMS = ["tile_m","tile_n","tile_k","tile_stages"]

def section(title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def summarize_schema(df: pd.DataFrame):
    section("SCHEMA & DTYPES (first 40)")
    print(df.dtypes.head(40))

    section("NULL COUNTS (top 30)")
    print(df.isna().sum().sort_values(ascending=False).head(30))

def summarize_categoricals(df: pd.DataFrame):
    section("CATEGORICAL UNIQUES")
    for c in CATEGORICALS:
        if c in df.columns:
            vals = df[c].dropna().unique().tolist()
            print(f"{c}: {sorted(vals)}  (nulls={df[c].isna().sum()})")

def summarize_metrics(df: pd.DataFrame):
    section("KEY METRIC RANGES")
    for c in KEY_METRICS:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            print(f"{c}: count={x.notna().sum()}  min={x.min()}  p50={x.median()}  max={x.max()}")

def coverage(df: pd.DataFrame):
    section("COVERAGE BY OPERATION")
    if "operation" in df.columns:
        print(df["operation"].value_counts(dropna=False))
    if "conv_role" in df.columns and "operation" in df.columns:
        print("\nconv2d by role:")
        print(df[df["operation"]=="conv2d"]["conv_role"].value_counts(dropna=False))

    # Problem-size availability
    section("PROBLEM SIZE COVERAGE (counts of non-nulls)")
    ps = df[PROBLEM_DIMS].notna().sum() if all(k in df.columns for k in PROBLEM_DIMS) else pd.Series(dtype=int)
    print(ps if not ps.empty else "No problem-size columns present.")

def sample_rows(df: pd.DataFrame):
    section("SAMPLE ROWS (first 12)")
    show = ["operation","Kernel Name"] + [c for c in KEY_METRICS if c in df.columns]
    show = [c for c in show if c in df.columns]
    print(df[show].head(12).to_string(index=False))

def tile_summaries(df: pd.DataFrame):
    section("TILE FEATURE SUMMARY")
    have = [c for c in TILE_DIMS if c in df.columns]
    if not have:
        print("No tile dims present.")
    else:
        print("non-null counts:")
        print(df[have].notna().sum())
        print("\ndescriptive stats:")
        print(df[have].describe().to_string())

    if "tile_shape_bucket" in df.columns:
        print("\ntile_shape_bucket counts:")
        print(df["tile_shape_bucket"].value_counts(dropna=False))

def leaders(df: pd.DataFrame):
    section("LEADERBOARDS")
    if "gpu__time_duration.sum" in df.columns:
        print("\nFASTEST (lowest gpu__time_duration.sum) top 10:")
        print(df.nsmallest(10, "gpu__time_duration.sum")[["operation","Kernel Name","gpu__time_duration.sum"]]
              .to_string(index=False))
    if "sm__throughput.avg.pct_of_peak_sustained_elapsed" in df.columns:
        print("\nHIGHEST THROUGHPUT top 10:")
        print(df.nlargest(10, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
              [["operation","Kernel Name","sm__throughput.avg.pct_of_peak_sustained_elapsed"]]
              .to_string(index=False))

def correlations(df: pd.DataFrame):
    section("CORRELATIONS (metrics + tile dims)")
    cols = [c for c in KEY_METRICS if c in df.columns] + [c for c in TILE_DIMS if c in df.columns]
    if not cols:
        print("No numeric columns to correlate.")
        return
    X = df[cols].copy()
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    corr = X.corr(numeric_only=True)
    print(corr.round(3).to_string())

def data_dictionary(df: pd.DataFrame, out_csv: Path):
    section("DATA DICTIONARY (writing CSV)")
    rows=[]
    for c in df.columns:
        s = df[c]
        miss = s.isna().mean()
        dtype = str(s.dtype)
        sample = s.dropna().iloc[0] if s.notna().any() else None
        rows.append({"column": c, "dtype": dtype, "missing_frac": round(miss,4), "sample": sample})
    dd = pd.DataFrame(rows).sort_values(["dtype","column"])
    dd.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} (rows: {len(dd)})")

def per_op_breakdowns(df: pd.DataFrame):
    section("PER-OP BREAKDOWNS (basic)")
    if "operation" not in df.columns:
        print("No 'operation' column.")
        return
    for op, g in df.groupby("operation"):
        print(f"\n-- {op} --")
        if "conv_role" in g.columns:
            print("roles:", g["conv_role"].value_counts(dropna=False).to_dict())
        # metric ranges
        for c in KEY_METRICS:
            if c in g.columns:
                x = pd.to_numeric(g[c], errors="coerce")
                print(f"{c}: n={x.notna().sum()} min={x.min()} p50={x.median()} max={x.max()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, help="Path to merged_all.parquet")
    ap.add_argument("--outdir", required=True, help="Where to write data_dictionary.csv")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.merged)

    section("BASIC INFO")
    print("rows:", len(df))

    summarize_schema(df)
    summarize_categoricals(df)
    summarize_metrics(df)
    coverage(df)
    sample_rows(df)
    tile_summaries(df)
    leaders(df)
    correlations(df)
    data_dictionary(df, outdir / "data_dictionary.csv")
    per_op_breakdowns(df)

if __name__ == "__main__":
    main()