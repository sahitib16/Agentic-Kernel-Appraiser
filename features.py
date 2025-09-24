from __future__ import annotations
import json
import re
import pandas as pd
from typing import Any, Dict

# helpers
VEC4 = re.compile(r"\b(float4|half4|int4)\b")

def _first_present(row: pd.Series, *names: str) -> Any:
    """Return first non-null among multiple possible column names."""
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return None

def _col(df: pd.DataFrame, *names: str) -> pd.Series:
    """Return a Series for the first name that exists in the dataframe, else a None-filled Series."""
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([None] * len(df), index=df.index)

def _parse_shape_cell(x: Any) -> list[int]:
    """Parse shape cell which may be list/tuple/JSON string/None into [d0,d1,d2]."""
    try:
        if isinstance(x, str):
            x = json.loads(x)
        if isinstance(x, (list, tuple)):
            out = list(x[:3])
            while len(out) < 3:
                out.append(0)
            return [int(v) for v in out]
    except Exception:
        pass
    return [0, 0, 0]

def _parse_ncu_cell(x: Any) -> Dict[str, float]:
    """Parse a potential NCU JSON blob into a small dict of numeric metrics (if present)."""
    if x is None:
        return {}
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except Exception:
            return {}
    if not isinstance(x, dict):
        return {}
    out: Dict[str, float] = {}
    for k in [
        "achieved_occupancy",
        "dram_throughput_gbps",
        "gld_efficiency",
        "gst_efficiency",
        "l2_hit_rate",
    ]:
        v = x.get(k, None)
        try:
            if v is not None:
                out[k] = float(v)
        except Exception:
            pass
    return out

def simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a minimal feature table + labels from a Sakana subset parquet.
    Works with either snake_case or CamelCase column names.
    """
    out = pd.DataFrame(index=df.index)

    # --- Code text
    code = _col(df, "cuda_code", "CUDA_Code").fillna("")
    out["code_len"] = code.str.len()
    out["uses_shared"] = code.str.contains("__shared__", regex=False).astype(int)
    out["uses_vec4"] = code.str.contains(VEC4).astype(int)
    out["uses_unroll"] = code.str.contains("#pragma unroll").astype(int)

    # --- Meta
    op_name = _col(df, "op_name", "Op_Name").fillna("unknown")
    level_id = _col(df, "level_id", "Level_ID").fillna("unknown")
    kernel_name = _col(df, "kernel_name", "Kernel_Name").fillna("")
    out["op_name"] = op_name
    out["level_id"] = level_id
    out["kernel_len"] = kernel_name.astype(str).str.len()

    # --- Shapes
    shapes = _col(df, "shape", "Shape")
    parsed_shapes = shapes.apply(_parse_shape_cell)
    out[["dim0", "dim1", "dim2"]] = pd.DataFrame(parsed_shapes.tolist(), index=df.index)

    # --- Runtimes & speedups (keep numeric)
    def _numcol(*names: str) -> pd.Series:
        return pd.to_numeric(_col(df, *names), errors="coerce")

    out["cuda_runtime"] = _numcol("cuda_runtime", "CUDA_Runtime")
    out["pytorch_native_runtime"] = _numcol("pytorch_native_runtime", "PyTorch_Native_Runtime")
    out["pytorch_compile_runtime"] = _numcol("pytorch_compile_runtime", "PyTorch_Compile_Runtime")
    out["cuda_speedup_native"] = _numcol("cuda_speedup_native", "CUDA_Speedup_Native").fillna(1.0).clip(lower=0)
    out["cuda_speedup_compile"] = _numcol("cuda_speedup_compile", "CUDA_Speedup_Compile").clip(lower=0)

    # --- NCU (optional)
    ncu_series = _col(df, "ncu_profile", "NCU_Profile")
    if ncu_series.notna().any():
        parsed = ncu_series.apply(_parse_ncu_cell)
        for k in ["achieved_occupancy", "dram_throughput_gbps", "gld_efficiency", "gst_efficiency", "l2_hit_rate"]:
            out[k] = parsed.apply(lambda d: d.get(k, None))

    # labels
    correct = _col(df, "correct", "Correct")
    out["label_correct"] = pd.to_numeric(correct, errors="coerce").fillna(0).astype("int8")
    # choose native speedup as primary label
    out["label_speedup"] = out["cuda_speedup_native"].fillna(1.0)
    # add other code-derived features
    out["count_sync"] = code.str.count(r"__syncthreads\s*\(")
    out["count_for"]  = code.str.count(r"\bfor\s*\(")
    out["count_if"]   = code.str.count(r"\bif\s*\(")

    return out