import re
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
FEATURES = DATA_DIR / "features_v0.parquet"
RAW_ALL  = DATA_DIR / "sakana_all.parquet"
RAW_LITE = DATA_DIR / "sakana_light.parquet"

def load_features() -> pd.DataFrame:
    if FEATURES.exists():
        df = pd.read_parquet(FEATURES)
        print(f"[load] features_v0: {len(df):,} rows")
        # ensure needed cols exist
        needed = ["label_correct", "label_speedup", "op_name", "level_id",
                  "uses_shared", "uses_unroll", "uses_vec4", "code_len"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"[warn] missing in features: {missing} — some summaries will be skipped.")
        return df
    # fallback: frame from raw (op_name/level_id only)
    src = RAW_LITE if RAW_LITE.exists() else RAW_ALL
    if src is None or not src.exists():
        raise FileNotFoundError("No features_v0.parquet and no raw parquet found.")
    df = pd.read_parquet(src)
    print(f"[load] raw: {src.name} {len(df):,} rows (features_v0 not found)")
    # standardize expected columns
    if "Correct" in df.columns and "correct" not in df.columns:
        df["correct"] = pd.to_numeric(df["Correct"], errors="coerce").fillna(0).astype("int8")
    if "CUDA_Speedup_Native" in df.columns and "cuda_speedup_native" not in df.columns:
        df["cuda_speedup_native"] = pd.to_numeric(df["CUDA_Speedup_Native"], errors="coerce").fillna(1.0)
    if "op_name" not in df.columns and "Op_Name" in df.columns:
        df["op_name"] = df["Op_Name"]
    if "level_id" not in df.columns and "Level_ID" in df.columns:
        df["level_id"] = df["Level_ID"]
    # minimal projection
    out = pd.DataFrame({
        "label_correct": pd.to_numeric(df.get("correct", 0), errors="coerce").fillna(0).astype("int8"),
        "label_speedup": pd.to_numeric(df.get("cuda_speedup_native", 1.0), errors="coerce").fillna(1.0),
        "op_name": df.get("op_name", "unknown"),
        "level_id": df.get("level_id", "unknown"),
    })
    return out

# --- Error helpers -------------------------------------------------

COMPILE_PATTERNS = [
    r"\berror: ", r"expected [^ ]+", r"undeclared identifier", r"no matching function",
    r"identifier .* is undefined", r"parse error", r"syntax error", r"cannot convert",
    r"invalid conversion", r"use of undeclared", r"unknown type name", r"not declared"
]
RUNTIME_PATTERNS = [
    r"illegal memory access", r"an illegal memory access was encountered",
    r"unspecified launch failure", r"misaligned address", r"device-side assert triggered",
    r"out of bounds", r"out-of-bounds", r"invalid argument", r"cudaError", r"segmentation fault"
]
VALIDATION_PATTERNS = [
    r"mismatch", r"max[_ ]?diff", r"nan", r"inf", r"does not match reference",
    r"tolerance", r"relative error", r"absolute error"
]

def categorize_error(msg: str) -> str:
    if not msg or not isinstance(msg, str):
        return "other/empty"
    low = msg.lower()
    if any(re.search(p, low) for p in COMPILE_PATTERNS):
        return "compile_error"
    if any(re.search(p, low) for p in RUNTIME_PATTERNS):
        return "runtime_error"
    if any(re.search(p, low) for p in VALIDATION_PATTERNS):
        return "validation_error"
    return "other/unknown"

def load_raw_for_errors() -> pd.DataFrame:
    src = RAW_ALL if RAW_ALL.exists() else RAW_LITE
    if src is None or not src.exists():
        raise FileNotFoundError("Raw parquet not found; run 01_load_sakana.py first.")
    df = pd.read_parquet(src)
    # unify names
    if "correct" not in df.columns and "Correct" in df.columns:
        df["correct"] = pd.to_numeric(df["Correct"], errors="coerce").fillna(0).astype("int8")
    if "error" not in df.columns and "Error" in df.columns:
        df["error"] = df["Error"]
    if "op_name" not in df.columns and "Op_Name" in df.columns:
        df["op_name"] = df["Op_Name"]
    if "level_id" not in df.columns and "Level_ID" in df.columns:
        df["level_id"] = df["Level_ID"]
    return df

# main

def main():
    # load features
    f = load_features()

    # global summaries
    print("\n== Label balance ==")
    print(f["label_correct"].value_counts(normalize=True).rename("fraction")*100)

    print("\n== Speedup stats (overall) ==")
    print(f["label_speedup"].describe())

    # per-op summaries
    if "op_name" in f.columns:
        per_op = f.groupby("op_name", observed=False).agg(
            n=("label_speedup","size"),
            correct_rate=("label_correct","mean"),
            med_speedup=("label_speedup","median"),
        ).sort_values("med_speedup", ascending=False)
        print("\n== Top ops by median speedup (n>=50) ==")
        print(per_op[per_op["n"]>=50].head(15))
        per_op.to_csv(DATA_DIR/"eda_per_op.csv", index=True)
    else:
        print("\n[info] op_name not present in features; skipping per-op summary.")

    if "level_id" in f.columns:
        per_lvl = f.groupby("level_id", observed=False).agg(
            n=("label_speedup","size"),
            correct_rate=("label_correct","mean"),
            med_speedup=("label_speedup","median"),
        ).sort_values("med_speedup", ascending=False)
        print("\n== Per-level summary ==")
        print(per_lvl)
        per_lvl.to_csv(DATA_DIR/"eda_per_level.csv", index=True)

    # feature correlations (binary flags + code_len)
    cols = [c for c in ["uses_shared","uses_unroll","uses_vec4","code_len"] if c in f.columns]
    if cols:
        print("\n== Feature prevalence by correctness (means for binaries; avg code_len) ==")
        print(f.groupby("label_correct", observed=False)[cols].mean())
        for c in [c for c in ["uses_shared","uses_vec4","uses_unroll"] if c in f.columns]:
            med = f.groupby(c, observed=False)["label_speedup"].median()
            print(f"\nMedian speedup by {c}:\n{med}")
        if "code_len" in f.columns:
            q = pd.qcut(f["code_len"], 4, duplicates="drop")
            print("\nMedian speedup by code_len quartile:")
            print(f.groupby(q, observed=False)["label_speedup"].median())
    else:
        print("\n[info] binary/code_len features not present; skipping correlations.")

    # correct-only 
    if "label_correct" in f.columns:
        fc = f[f["label_correct"] == 1].copy()
        print(f"\n== Correct-only subset: {len(fc):,} rows ==")

        for c in [c for c in ["uses_shared","uses_vec4","uses_unroll"] if c in fc.columns]:
            med = fc.groupby(c, observed=False)["label_speedup"].median()
            print(f"\n[correct-only] Median speedup by {c}:\n{med}")

        if "code_len" in fc.columns:
            q = pd.qcut(fc["code_len"], 4, duplicates="drop")
            print("\n[correct-only] Median speedup by code_len quartile:")
            print(fc.groupby(q, observed=False)["label_speedup"].median())

        # per-op stratification (correct-only) for shared and vec4
        if "op_name" in fc.columns:
            if "uses_shared" in fc.columns:
                po_shared = (
                    fc.groupby(["op_name","uses_shared"], observed=False)["label_speedup"]
                      .median().unstack(fill_value=0)
                )
                for col in [0,1]:
                    if col not in po_shared.columns:
                        po_shared[col] = 0.0
                po_shared["delta_shared"] = po_shared.get(1,0.0) - po_shared.get(0,0.0)
                if "uses_vec4" in fc.columns:
                    po_vec4 = (
                        fc.groupby(["op_name","uses_vec4"], observed=False)["label_speedup"]
                          .median().unstack(fill_value=0)
                    )
                    for col in [0,1]:
                        if col not in po_vec4.columns:
                            po_vec4[col] = 0.0
                    po_vec4["delta_vec4"] = po_vec4.get(1,0.0) - po_vec4.get(0,0.0)
                else:
                    po_vec4 = pd.DataFrame()
                counts = fc.groupby("op_name", observed=False).size().rename("n_correct")
                po_shared = po_shared.join(counts, how="left")
                if not po_vec4.empty:
                    po_vec4 = po_vec4.join(counts, how="left")

                # save CSVs
                po_shared.to_csv(DATA_DIR/"eda_correct_only_shared.csv")
                if not po_vec4.empty:
                    po_vec4.to_csv(DATA_DIR/"eda_correct_only_vec4.csv")
                min_n = 30
                print(f"\n[correct-only] Top ops where shared helps (delta_shared>0, n_correct≥{min_n}):")
                if "delta_shared" in po_shared.columns:
                    print(po_shared[po_shared["n_correct"]>=min_n]
                          .sort_values("delta_shared", ascending=False)
                          .head(10)[["n_correct","delta_shared"]])
                if not po_vec4.empty and "delta_vec4" in po_vec4.columns:
                    print(f"\n[correct-only] Top ops where vec4 helps (delta_vec4>0, n_correct≥{min_n}):")
                    print(po_vec4[po_vec4["n_correct"]>=min_n]
                          .sort_values("delta_vec4", ascending=False)
                          .head(10)[["n_correct","delta_vec4"]])
                if not po_vec4.empty:
                    deltas = (po_shared[["delta_shared","n_correct"]]
                              .join(po_vec4[["delta_vec4"]], how="outer"))
                    deltas.to_csv(DATA_DIR/"eda_correct_only_deltas.csv")
            else:
                print("\n[correct-only] uses_shared not present; skipping per-op stratification.")
        else:
            print("\n[correct-only] op_name not present; skipping per-op stratification.")
    else:
        print("\n[info] label_correct missing; cannot compute correct-only stats.")

    # error analysis
    raw = load_raw_for_errors()
    failures = raw[raw["correct"] == 0].copy()
    print(f"\n== Failures: {len(failures):,} rows ==")

    if "error" in failures.columns:
        failures["error_cat"] = failures["error"].apply(categorize_error)
        cat_counts = failures["error_cat"].value_counts(dropna=False)
        cat_pct = (cat_counts / len(failures) * 100).round(2)
        print("\n== Error category breakdown (failures only) ==")
        print(pd.DataFrame({"count": cat_counts, "pct": cat_pct}))
        samples = []
        for cat in ["compile_error","runtime_error","validation_error","other/unknown","other/empty"]:
            ex = failures.loc[failures["error_cat"]==cat, ["op_name","level_id","error"]].dropna().head(5)
            ex["error_cat"] = cat
            samples.append(ex)
        if samples:
            out = pd.concat(samples, ignore_index=True)
            out.to_csv(DATA_DIR/"eda_error_samples.csv", index=False)
            print("\nSaved sample error messages to data/eda_error_samples.csv")
    else:
        print("[info] 'error' column not present in raw; cannot analyze failure modes.")

    head_cols = [c for c in ["op_name","level_id","uses_shared","uses_unroll","uses_vec4",
                              "code_len","label_correct","label_speedup"] if c in f.columns]
    f[head_cols].head(200).to_csv(DATA_DIR/"eda_head_sample.csv", index=False)
    print("\nSaved: eda_per_op.csv, eda_per_level.csv (if available), "
          "eda_correct_only_shared.csv, eda_correct_only_vec4.csv, eda_correct_only_deltas.csv, "
          "eda_error_samples.csv, eda_head_sample.csv")

if __name__ == "__main__":
    main()