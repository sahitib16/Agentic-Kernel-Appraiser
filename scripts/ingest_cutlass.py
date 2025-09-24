import json, subprocess, time, uuid, re
from io import StringIO
from pathlib import Path
import pandas as pd
import os
from ncu_utils import ncu_profile_exec
NCU_TMP = Path("data/ncu_tmp")
USE_NCU = bool(int(os.environ.get("AKA_USE_NCU", "0")))

DATA         = Path("data"); DATA.mkdir(exist_ok=True)
MANIFEST     = DATA / "cutlass_manifest.csv"
OUT_PARQUET  = DATA / "runs.parquet"
CUTLASS_BIN  = Path("tools/cutlass/build/tools/profiler/cutlass_profiler")

CSV_HEADER_PREFIX = "Problem,Provider,OperationKind,Operation,Disposition"

def _run(cmd: list[str]) -> tuple[str, str, int]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.stdout, p.stderr, p.returncode

def _extract_csv_text(stdout: str, stderr: str) -> str | None:
    # CUTLASS prints the table
    for blob in (stdout, stderr, stdout + "\n" + stderr):
        if CSV_HEADER_PREFIX in blob:
            start = blob.find(CSV_HEADER_PREFIX)
            if start != -1:
                tail = blob[start:].strip()
                lines = [ln for ln in tail.splitlines() if "," in ln]
                if lines:
                    return "\n".join(lines)
    return None

def run_profiler(op: str, size: list[int], dtype: str, extra: str) -> pd.DataFrame:
    # verification default (ON) 
    args = [
        str(CUTLASS_BIN),
        f"--operation={op}",
        f"--dtype={dtype}",
        "--benchmark-enabled=true",
        "--profiling-iterations=1",
        "--min-iterations=1",
        "--warmup-iterations=0",
        "--report-not-run=true",
        "--report-env=false",
        "--verbose=true",
    ]

    if op.lower() == "gemm":
        m, n, k = size
        args += [
            f"--m={m}", f"--n={n}", f"--k={k}",
            f"--A={dtype}:column", f"--B={dtype}:column",
            f"--C={dtype}:column", f"--D={dtype}:column",
            "--alpha=1", "--beta=0",
            f"--accum={dtype}",
            "--op_class=simt",
        ]
    elif op.lower() == "conv2d":
        n,c,h,w = size
        args += [
            "--mode=Fprop",
            f"--n={n}", f"--h={h}", f"--w={w}", f"--c={c}",
            "--k=64", "--r=3", "--s=3",
            "--pad_h=1", "--pad_w=1", "--stride_h=1", "--stride_w=1",
            "--op_class=simt",
        ]

    if extra:
        args += extra.split()

    print("[cmd]", " ".join(args))
    out, err, code = _run(args)

    csv_text = _extract_csv_text(out, err)
    if not csv_text:
        print("No CSV block detected.\nstdout(head):\n", "\n".join(out.splitlines()[:40]),
              "\nstderr(head):\n", "\n".join(err.splitlines()[:40]))
        return pd.DataFrame()

    try:
        df = pd.read_csv(StringIO(csv_text))
    except Exception:
        lines = csv_text.splitlines()
        header = lines[0]
        rows = [ln for ln in lines[1:] if re.match(r"^\d+,", ln)]
        df = pd.read_csv(StringIO("\n".join([header] + rows)))

    # annotate
    df["source"] = "cutlass"
    df["op_name"] = op
    df["dtype"] = dtype
    df["shape"] = str(size)
    df["ts"] = time.time()
    # normalize a couple common columns 
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "operation": rename[c] = "impl"
        if lc == "runtime":   rename[c] = "runtime_ms"
    return df.rename(columns=rename)

def main():
    if not CUTLASS_BIN.exists():
        raise FileNotFoundError(f"cutlass_profiler not found at {CUTLASS_BIN}")
    mf = pd.read_csv(MANIFEST)
    dfs = []
    for _, row in mf.iterrows():
        op     = row["op"]
        sizes  = json.loads(row["problem_sizes"])
        dtype  = row.get("dtype", "f32")
        extra  = row.get("extra_args", "")
        for sz in sizes:
            df = run_profiler(op, sz, dtype, extra)
            if not df.empty:
                dfs.append(df)
    if not dfs:
        print("No rows produced.")
        return
    out_df = pd.concat(dfs, ignore_index=True)
    if OUT_PARQUET.exists():
        prev = pd.read_parquet(OUT_PARQUET)
        out_df = pd.concat([prev, out_df], ignore_index=True)
    if USE_NCU:
        enriched = []
        for i, r in out_df.iterrows():
            op = r["op_name"]; dtype = r.get("dtype","f32")
            # Problem size comes from the 'shape' string; turn back to list[int]
            shape = eval(r["shape"]) if isinstance(r["shape"], str) else r["shape"]
            args = [f"--operation={op}", f"--dtype={dtype}",
                    "--benchmark-enabled=true", "--profiling-iterations=1", "--min-iterations=1", "--warmup-iterations=0",
                    "--report-not-run=true", "--report-env=false", "--verbose=false",
                    f"--kernels={r.get('impl', r.get('Operation',''))}"]
            if op=="gemm":
                m,n,k = shape
                args += [f"--m={m}", f"--n={n}", f"--k={k}",
                        f"--A={dtype}:column", f"--B={dtype}:column", f"--C={dtype}:column", f"--D={dtype}:column",
                        "--alpha=1", "--beta=0", f"--accum={dtype}", "--op_class=simt"]
            elif op=="conv2d":
                n,c,h,w = shape
                args += ["--mode=Fprop", f"--n={n}", f"--h={h}", f"--w={w}", f"--c={c}",
                        "--k=64","--r=3","--s=3","--pad_h=1","--pad_w=1","--stride_h=1","--stride_w=1","--op_class=simt"]

            prof = ncu_profile_exec(str(CUTLASS_BIN), args, NCU_TMP, f"data/ncu_{op}_{i}")
            if prof:
                for k,v in prof.items(): out_df.loc[i, k] = v
    print(out_df.columns)
    out_df.to_parquet(OUT_PARQUET, index=False)
    print("Saved", OUT_PARQUET, "rows:", len(out_df))

if __name__ == "__main__":
    main()