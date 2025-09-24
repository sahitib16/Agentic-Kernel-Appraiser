import os, subprocess, pandas as pd
from pathlib import Path
from uuid import uuid4

NCU_METRICS = [
  "sm__warps_active.avg.pct_of_peak_sustained_active",
  "sm__throughput.avg.pct_of_peak_sustained_elapsed",
  "dram__throughput.avg.pct_of_peak_sustained_elapsed",
  "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second",
]

def ncu_profile_exec(exe: str, args_list: list[str], tmpdir: Path, export_prefix: str) -> dict:
    tmpdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TMPDIR"] = str(tmpdir)  

    uid = uuid4().hex[:8]
    rep = tmpdir / f"{Path(export_prefix).stem}_{uid}.ncu-rep"

    try:
        p = subprocess.run(
            ["ncu","--export",str(rep),"--target-processes","all",
             "--metrics",",".join(NCU_METRICS), exe, *args_list],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, timeout=180
        )
    except subprocess.TimeoutExpired:
        return {}

    if p.returncode != 0 or not rep.exists():
        # no report -> no metrics
        return {}

    # convert to CSV in tmpdir
    out = subprocess.run(["ncu","--import",str(rep),"--page","raw","--csv"],
                         stdout=subprocess.PIPE, text=True)
    csv_text = out.stdout
    if not csv_text.strip():
        return {}

    # parse
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(csv_text), dtype=str)
    except Exception:
        return {}

    if "Kernel Name" not in df.columns:
        return {}

    keep = ["Kernel Name"] + [m for m in NCU_METRICS if m in df.columns]
    df = df[keep].copy()
    df = df[df["Kernel Name"].notna() & (df["Kernel Name"].str.len()>0)]
    for c in keep[1:]:
        df[c] = pd.to_numeric(df[c].str.replace(",",""), errors="coerce")
    s = df.groupby("Kernel Name").median()
    if s.empty: return {}

    outd = {}
    if NCU_METRICS[0] in s.columns: outd["achieved_occupancy"] = s.iloc[0][NCU_METRICS[0]]
    if NCU_METRICS[1] in s.columns: outd["sm_throughput_pct"]  = s.iloc[0][NCU_METRICS[1]]
    if NCU_METRICS[2] in s.columns: outd["dram_throughput_pct"]= s.iloc[0][NCU_METRICS[2]]
    if NCU_METRICS[3] in s.columns: outd["gld_gbps"]           = s.iloc[0][NCU_METRICS[3]]/1e9
    return outd