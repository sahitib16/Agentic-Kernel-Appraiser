import os, argparse, pandas as pd
from pathlib import Path

def normalize_kernel_name(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.lower().str.strip()
    s = s.str.replace(r"\balign\d+\b", "", regex=True)
    s = s.str.replace(r"__+", "_", regex=True)
    s = s.str.replace(r"\s+", "", regex=True)
    return s.str.strip("_")

def ensure_kernel_norm(df: pd.DataFrame, candidates=("kernel_name_norm","kernel_name","Kernel Name","sakana_kernel_name")) -> pd.DataFrame:
    """
    Ensure df has a 'kernel_name_norm' column by normalizing the first candidate that exists.
    No-op if it already exists.
    """
    if "kernel_name_norm" in df.columns:
        return df
    for c in candidates:
        if c in df.columns:
            df = df.copy()
            df["kernel_name_norm"] = normalize_kernel_name(df[c])
            return df
    # If nothing to normalize, create an empty column so downstream doesn't crash
    df = df.copy()
    df["kernel_name_norm"] = ""
    return df

def build_labels(manifest: pd.DataFrame) -> pd.DataFrame:
    m = manifest.copy()
    m["compile_ok"] = ((m.get("return_code") == 0) | (m.get("status") == "ok")).fillna(False)
    m["ncu_ok"] = ((m.get("ncu_rows", 0).fillna(0) >= 1) & (m.get("ncu_read_error").isna())).fillna(False)
    m["run_ok"] = (m["compile_ok"] & m["ncu_ok"]).astype(int)
    m["label_fail"] = (1 - m["run_ok"]).astype(int)
    keep = ["run_dir","operation","kernel_name","compile_ok","ncu_ok","run_ok","label_fail",
            "status","return_code","ncu_rows","ncu_read_error"]
    keep = [c for c in keep if c in m.columns]
    return m[keep]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--sakana", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(args.manifest)
    metrics  = pd.read_parquet(args.metrics)
    feats    = pd.read_parquet(args.features)
    sakana   = pd.read_parquet(args.sakana)

    # Build labels
    labels = build_labels(manifest)

    # Ensure each table has kernel_name_norm (derive from whatever exists)
    feats   = ensure_kernel_norm(feats,   candidates=("kernel_name","Kernel Name"))
    labels  = ensure_kernel_norm(labels,  candidates=("kernel_name","Kernel Name"))
    metrics = ensure_kernel_norm(metrics, candidates=("Kernel Name","kernel_name"))
    sakana  = ensure_kernel_norm(sakana,  candidates=("kernel_name_norm","kernel_name"))

    # Merge your three tables (features ⨝ labels ⨝ metrics) by run_dir first
    merged = (feats
              .merge(labels.drop(columns=[c for c in ["kernel_name"] if c in labels.columns]),
                     on="run_dir", how="left")
              .merge(metrics[["run_dir","Kernel Name","ncu_kernel_symbol"]], on="run_dir", how="left"))

    # Join to Sakana on normalized kernel name
    merged = merged.merge(
        sakana[["kernel_name_norm","kernel_name","ncu_profile","correct","error"]]
              .rename(columns={"kernel_name":"sakana_kernel_name",
                               "ncu_profile":"sakana_ncu_profile",
                               "error":"sakana_error"}),
        on="kernel_name_norm", how="left"
    )

    out_full  = os.path.join(args.outdir, "merged_with_sakana.parquet")
    merged.to_parquet(out_full, index=False)

    train_cols = [
        "run_ok","compile_ok","ncu_ok","label_fail",
        "operation","kernel_name","kernel_name_norm",
        "math_mode","dtype","layout","conv_role",
        "tile_m","tile_n","tile_k","tile_stages","tile_shape_bucket",
        "has_unity_stride","is_depthwise",
        "gpu__time_duration.sum","sm__throughput.avg.pct_of_peak_sustained_elapsed","dram__bytes.sum",
        "sakana_kernel_name","sakana_ncu_profile","correct","sakana_error"
    ]
    train_cols = [c for c in train_cols if c in merged.columns]
    merged[train_cols].to_parquet(os.path.join(args.outdir,"train_table.parquet"), index=False)

    print("Wrote:")
    print(" -", out_full)
    print(" -", os.path.join(args.outdir,"train_table.parquet"))
    print("Rows:", len(merged))

if __name__ == "__main__":
    main()