# scripts/prepare_splits.py
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse, os
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to train_table_v5.parquet")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} cols")

    # drop rows without labels or code
    df = df[df["correct"].notna()]
    df = df[df["code_text"].notna()]
    print(f"After filtering: {len(df):,} rows")

    # stratified split (preserve correct/incorrect balance)
    train_df, temp = train_test_split(
        df, test_size=0.3, stratify=df["correct"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp, test_size=0.5, stratify=temp["correct"], random_state=42
    )

    # prepare code-only test view (drop numeric / runtime columns)
    runtime_cols = [c for c in df.columns if "gpu__" in c or "dram__" in c or "sm__" in c]
    numeric_cols = ["tile_m","tile_n","tile_k","tile_stages"]
    drop_for_test = list(set(runtime_cols + numeric_cols))

    code_test = test_df.drop(columns=drop_for_test, errors="ignore")
    # Keep only code, simple categorical context, and label
    keep = ["code_text","op_kind","dtype","layout","math_mode","conv_role","correct"]
    code_test = code_test[[c for c in keep if c in code_test.columns]]

    # Save
    train_df.to_parquet(os.path.join(args.outdir,"train_split.parquet"), index=False)
    val_df.to_parquet(os.path.join(args.outdir,"val_split.parquet"), index=False)
    code_test.to_parquet(os.path.join(args.outdir,"test_split.parquet"), index=False)

    # Print summary
    for name, d in [("train",train_df),("val",val_df),("test",code_test)]:
        print(f"\n{name.upper()} → {len(d):,} rows, correct=1 fraction={d['correct'].mean():.3f}")
        print(f"  columns: {', '.join(d.columns[:10])} ...")

if __name__ == "__main__":
    main()