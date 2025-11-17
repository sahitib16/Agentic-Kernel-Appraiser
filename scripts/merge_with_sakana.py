# scripts/merge_with_sakana.py
import os
import re
import argparse
from pathlib import Path
import pandas as pd


def normalize_kernel_name(s: pd.Series) -> pd.Series:
    """Canonicalize kernel names for joining."""
    out = s.fillna("").astype(str).str.lower().str.strip()
    out = out.str.replace(r"\balign\d+\b", "", regex=True)
    out = out.str.replace(r"\s+", "", regex=True)
    out = out.str.replace(r"__+", "_", regex=True)
    return out.str.strip("_")


def combine_first(df: pd.DataFrame, base: str, prefer: str, fallback: str) -> pd.DataFrame:
    """
    Create/replace df[base] by taking non-null from df[prefer], then from df[fallback].
    Drops the source columns if present.
    """
    had_any = False
    if prefer in df.columns or fallback in df.columns:
        had_any = True
        df[base] = df.get(prefer)
        if fallback in df.columns:
            df[base] = df[base].where(df[base].notna(), df[fallback])
        for c in (prefer, fallback):
            if c in df.columns:
                df.drop(columns=[c], inplace=True)
    if not had_any and base not in df.columns:
        df[base] = None
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=False, help="(optional) for labels; not required by this script")
    ap.add_argument("--metrics", required=False, help="(optional) for labels; not required by this script")
    ap.add_argument("--features", required=True, help="CUTLASS features parquet (may include ptx_code)")
    ap.add_argument("--sakana",   required=True, help="Sakana enriched parquet (cuda_code/correct/op_name/etc.)")
    ap.add_argument("--outdir",   required=True)
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # ---------------- CUTLASS side ----------------
    cut = pd.read_parquet(args.features).copy()
    # Expect columns like: kernel_name, operation, ptx_code, dtype, layout, math_mode, conv_role, tile_*, metrics...
    cut["kernel_name_norm"] = normalize_kernel_name(cut.get("kernel_name"))

    # Prefix everything to avoid collisions, but keep normalized key
    cut_pref = cut.add_prefix("cut_").rename(columns={"cut_kernel_name_norm": "kernel_name_norm"})

    # ---------------- Sakana side ----------------
    sak = pd.read_parquet(args.sakana).copy()
    # Common columns in your enriched file: kernel_name, op_name/op_name-like, correct, cuda_code, error, ncu_profile, etc.
    if "kernel_name_norm" not in sak.columns:
        sak["kernel_name_norm"] = normalize_kernel_name(sak.get("kernel_name"))

    sak_pref = sak.add_prefix("sak_").rename(columns={"sak_kernel_name_norm": "kernel_name_norm"})

    # Construct a generic Sakana "name" column from whatever exists
    name_candidates = [c for c in ["sak_kernel_name", "sak_name", "sak_op_name"] if c in sak_pref.columns]
    if name_candidates:
        # left-to-right first non-null
        sak_pref["sak_any_name"] = sak_pref[name_candidates].bfill(axis=1).iloc[:, 0]
    else:
        sak_pref["sak_any_name"] = None

    # ---------------- Merge (outer) ----------------
    merged = cut_pref.merge(sak_pref, on="kernel_name_norm", how="outer", validate="m:m")

    # dataset-source flags
    merged["is_cutlass"] = merged.get("cut_run_dir").notna().astype("Int8") if "cut_run_dir" in merged.columns else 0
    if "sak_any_name" in merged.columns:
        merged["is_sakana"] = merged["sak_any_name"].notna().astype("Int8")
    else:
        sak_cols = [c for c in merged.columns if c.startswith("sak_")]
        merged["is_sakana"] = (merged[sak_cols].notna().any(axis=1)).astype("Int8") if sak_cols else 0

    # unified kernel_name
    if "sak_any_name" in merged.columns:
        merged = combine_first(merged, "kernel_name", "cut_kernel_name", "sak_any_name")
    else:
        merged = combine_first(merged, "kernel_name", "cut_kernel_name", "sak_op_name")

    # unified operation (prefer CUTLASS operation; fallback to Sakana op_name)
    if "sak_op_name" in merged.columns:
        merged = combine_first(merged, "operation", "cut_operation", "sak_op_name")
    else:
        merged = combine_first(merged, "operation", "cut_operation", "operation")

    # correctness: prefer Sakana's label if present; force 1 for CUTLASS rows (these ran)
    merged["correct"] = None
    if "sak_correct" in merged.columns:
        merged["correct"] = pd.to_numeric(merged["sak_correct"], errors="coerce")
    merged.loc[merged["is_cutlass"] == 1, "correct"] = 1
    merged["correct"] = merged["correct"].fillna(0).astype("Int8")

    # Carry code fields
    merged["cuda_code"] = merged.get("sak_cuda_code")  # Sakana CUDA source if present
    merged["ptx_code"]  = merged.get("cut_ptx_code")   # CUTLASS PTX if present

    # unified code_text: prefer cuda_code (source), else PTX
    merged["code_text"] = merged["cuda_code"]
    need_fill = merged["code_text"].isna()
    if "ptx_code" in merged.columns:
        merged.loc[need_fill, "code_text"] = merged.loc[need_fill, "ptx_code"]

    # Basic numeric metrics from CUTLASS (if present)
    for col in [
        "gpu__time_duration.sum",
        "dram__bytes.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    ]:
        ccol = f"cut_{col}"
        if ccol in merged.columns:
            merged[col] = pd.to_numeric(merged[ccol], errors="coerce")

    # Static descriptors (prefer CUTLASS; fallback Sakana if present)
    for base, csrc, ssrc in [
        ("dtype", "cut_dtype", "sak_dtype"),
        ("layout", "cut_layout", "sak_layout"),
        ("math_mode", "cut_math_mode", "sak_math_mode"),
        ("conv_role", "cut_conv_role", "sak_conv_role"),
        ("tile_m", "cut_tile_m", None),
        ("tile_n", "cut_tile_n", None),
        ("tile_k", "cut_tile_k", None),
        ("tile_stages", "cut_tile_stages", None),
        ("tile_shape_bucket", "cut_tile_shape_bucket", None),
        ("is_depthwise", "cut_is_depthwise", None),
        ("has_unity_stride", "cut_has_unity_stride", None),
    ]:
        if csrc in merged.columns or (ssrc and ssrc in merged.columns):
            prefer = csrc
            fallback = ssrc if ssrc else csrc  # no good fallback; keep csrc if missing
            merged = combine_first(merged, base, prefer, fallback)

    # Write full merged
    out_full = os.path.join(args.outdir, "merged_with_sakana.parquet")
    merged.to_parquet(out_full, index=False)

    # Compact train table
    keep = [
        "kernel_name_norm","kernel_name","operation",
        "is_cutlass","is_sakana","correct",
        # static
        "dtype","layout","math_mode","conv_role",
        "tile_m","tile_n","tile_k","tile_stages","tile_shape_bucket",
        "is_depthwise","has_unity_stride",
        # metrics
        "gpu__time_duration.sum","dram__bytes.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed","sm__warps_active.avg.pct_of_peak_sustained_active",
        # code
        "code_text","ptx_code","cuda_code",
        # helpful ids / fields
        "cut_run_dir","sak_level_id","sak_task_id","sak_error",
    ]
    keep = [c for c in keep if c in merged.columns]
    train = merged[keep].copy()
    out_train = os.path.join(args.outdir, "train_table.parquet")
    train.to_parquet(out_train, index=False)

    print("Wrote:")
    print(" -", out_full)
    print(" -", out_train)
    print("Rows:", len(merged))


if __name__ == "__main__":
    main()