# scripts/enrich_sakana_ncu.py
# Purpose: parse Sakana's ncu_profile JSON (if present), extract Nsight-like metrics,
# normalize kernel_name to a canonical key, and write an enriched parquet.

import os
import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# ---------- helpers ----------

def normalize_kernel_name_series(s: pd.Series) -> pd.Series:
    """Lowercase, strip, remove spaces, collapse underscores, drop align tags."""
    s = s.fillna("").astype(str).str.lower().str.strip()
    s = s.str.replace(r"\balign\d+\b", "", regex=True)
    s = s.str.replace(r"\s+", "", regex=True)      # remove spaces
    s = s.str.replace(r"__+", "_", regex=True)     # collapse consecutive underscores
    return s.str.strip("_")

def safe_json_load(x: Any):
    """Robustly parse JSON-ish content from Sakana's ncu_profile column."""
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, str):
        t = x.strip()
        if not t:
            return None
        # try as-is
        try:
            return json.loads(t)
        except Exception:
            # fallback: single quotes -> double quotes
            try:
                return json.loads(t.replace("'", '"'))
            except Exception:
                return None
    return None

def flatten(d: Any, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dict/list into dotted keys (a.b.c[3])."""
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            out.update(flatten(v, key, sep))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            key = f"{parent_key}[{i}]"
            out.update(flatten(v, key, sep))
    else:
        out[parent_key] = d
    return out

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def coerce_number(x) -> float | None:
    """Turn '296,384', '44.58 %', '123 us' into a float; return None if not numeric."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.replace(",", " ").strip()
        m = _NUM_RE.search(s)
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None

# Map common Sakana flattened keys → standardized columns.
# We match by substring on lowercased flattened keys.
PATTERNS = (
    # GPU time (microseconds)
    ("gpu__time_duration.sum", "sak_gpu_time_us"),
    ("gpu__time_duration",     "sak_gpu_time_us"),
    ("gpu_time_us",            "sak_gpu_time_us"),

    # DRAM traffic (bytes)
    ("dram__bytes.sum",        "sak_dram_bytes"),
    ("dram_bytes",             "sak_dram_bytes"),
    ("dram__bytes_read.sum",   "sak_dram_bytes_read"),
    ("dram__bytes_write.sum",  "sak_dram_bytes_write"),

    # L2 traffic (bytes)
    ("lts__t_bytes.sum",       "sak_lts_t_bytes"),
    ("l2_bytes",               "sak_lts_t_bytes"),

    # SM utilization (percent of peak)
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "sak_sm_throughput_pct"),
    ("sm_throughput_pct",      "sak_sm_throughput_pct"),

    ("sm__warps_active.avg.pct_of_peak_sustained_active", "sak_sm_warps_active_pct"),
    ("sm_warps_active_pct",    "sak_sm_warps_active_pct"),

    ("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", "sak_sm_pipe_tensor_cycles_active_pct"),
    ("sm_pipe_tensor_cycles_active_pct", "sak_sm_pipe_tensor_cycles_active_pct"),
)

DEFAULT_OUT = {
    "sak_gpu_time_us": None,
    "sak_dram_bytes": None,
    "sak_dram_bytes_read": None,
    "sak_dram_bytes_write": None,
    "sak_lts_t_bytes": None,
    "sak_sm_throughput_pct": None,
    "sak_sm_warps_active_pct": None,
    "sak_sm_pipe_tensor_cycles_active_pct": None,
}

def extract_from_flat(flat: Dict[str, Any]) -> Dict[str, float | None]:
    """Find the first match for each metric pattern and coerce to float."""
    out = DEFAULT_OUT.copy()
    if not flat:
        return out
    items = [(k.lower(), v) for k, v in flat.items()]
    for key_l, val in items:
        for pat, col in PATTERNS:
            if out[col] is None and pat in key_l:
                out[col] = coerce_number(val)
    return out

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Enrich Sakana dataset by parsing ncu_profile into numeric columns.")
    ap.add_argument("--in", dest="inp", required=True, help="Path to data/sakana_all.parquet")
    ap.add_argument("--outdir", required=True, help="Output dir, e.g. data/derived/sakana")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.inp)

    # Parse ncu_profile (JSON-ish) → flattened dict → extract numeric metrics
    # We'll iterate; 30k rows is fine performance-wise for this pass.
    records = []
    for _, row in df.iterrows():
        prof_raw = row.get("ncu_profile", None)
        prof_json = safe_json_load(prof_raw)
        flat = flatten(prof_json) if prof_json is not None else {}
        metrics = extract_from_flat(flat)

        rec = dict(row)
        rec.update(metrics)
        records.append(rec)

    out = pd.DataFrame(records)

    # Ensure a normalized join key exists
    if "kernel_name_norm" not in out.columns:
        src = out["kernel_name"] if "kernel_name" in out.columns else pd.Series([""] * len(out))
        out["kernel_name_norm"] = normalize_kernel_name_series(src)

    out_path = os.path.join(args.outdir, "sakana_enriched.parquet")
    out.to_parquet(out_path, index=False)
    print(f"Wrote: {out_path} rows: {len(out)}")

if __name__ == "__main__":
    main()