#!/usr/bin/env python3
import os, re, json, shutil, subprocess, tempfile
from pathlib import Path

def sh(cmd, timeout=1200, to_file=None):
    if to_file:
        with open(to_file,"w") as f:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True, timeout=timeout)
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)

def which_profiler(cli=None):
    cands=[]
    if cli: cands.append(cli)
    if os.getenv("CUTLASS_PROFILER"): cands.append(os.getenv("CUTLASS_PROFILER"))
    if os.getenv("CUTLASS_HOME"):
        ch=os.getenv("CUTLASS_HOME")
        cands += [f"{ch}/build/tools/profiler/cutlass_profiler", f"{ch}/tools/profiler/cutlass_profiler"]
    wp=shutil.which("cutlass_profiler")
    if wp: cands.append(wp)
    for p in cands:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return os.path.abspath(p)
    raise FileNotFoundError("cutlass_profiler not found")

def detect_arch():
    try:
        out = sh(["nvidia-smi","--query-gpu=compute_cap","--format=csv,noheader"], timeout=5).stdout.strip()
        if out:
            maj,min = out.split(".")
            return f"sm{maj}{min}", f"{maj}{min}", f"{maj}{min}"
    except Exception: pass
    return "", "50", "1024"

def run_to_lines(cmd):
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tf:
        path = Path(tf.name)
    try:
        p = sh(cmd, to_file=path)
        if p.returncode != 0: raise RuntimeError(p.stderr.strip() or f"rc={p.returncode}")
        return [ln.rstrip("\n") for ln in path.read_text(errors="ignore").splitlines()]
    finally:
        try: path.unlink()
        except Exception: pass

def parse_jsonl_or_tsv(lines):
    rows=[]
    # try JSONL
    tmp=[]; json_ok=True
    for ln in lines:
        if not ln.strip(): continue
        try: tmp.append(json.loads(ln))
        except Exception: json_ok=False; tmp=[]; break
    if json_ok and tmp: return tmp
    # TSV-ish (header + rows)
    header=None
    for ln in lines:
        parts=[c.strip() for c in re.split(r"\t+| {2,}", ln) if c.strip()]
        if not parts: continue
        if header is None:
            if len(parts)>1: header=parts
            continue
        if len(parts)==len(header): rows.append(dict(zip(header, parts)))
    return rows

def parse_report_blocks(lines):
    """Fallback: extract OperationKind + Operation from human report."""
    rows=[]; op=None
    for ln in lines:
        m=re.match(r'^\s*OperationKind:\s*(\S+)', ln)
        if m:
            op=m.group(1).strip().lower()
            continue
        m=re.match(r'^\s*Operation:\s*(\S.*\S)', ln)
        if m and op:
            name=m.group(1).strip()
            rows.append({"operation":op, "name":name})
            op=None
    return rows

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--profiler", type=str, default=None)
    ap.add_argument("--out", type=str, default="data/ncu/kernel_catalog.jsonl")
    args=ap.parse_args()

    prof = which_profiler(args.profiler)
    ARCH, MINCC, MAXCC = detect_arch()

    # 1) full enumerate with arch clamp
    cmd = [prof, "--enumerate", "--library", f"--min_cc={MINCC}", f"--max_cc={MAXCC}"]
    if ARCH: cmd.append(f"--arch={ARCH}")
    lines = run_to_lines(cmd)

    rows = parse_jsonl_or_tsv(lines)
    if not rows:
        # maybe it's a report; try block parser
        rows = parse_report_blocks(lines)

    # 2) if still low, add per-op (append)
    if len(rows) < 500:  # heuristic; adjust if needed
        help_txt = sh([prof,"--help"]).stdout
        m = re.search(r"--operation[^=<]*[=<]\s*<([^>]+)>", help_txt)
        ops = [t.strip().lower() for t in (m.group(1).split("|") if m else []) if re.fullmatch(r"[a-z_][a-z0-9_]*", t.strip().lower())]
        for op in ops:
            cmd = [prof, f"--operation={op}", "--enumerate", f"--min_cc={MINCC}", f"--max_cc={MAXCC}"]
            if ARCH: cmd.append(f"--arch={ARCH}")
            more = run_to_lines(cmd)
            parsed = parse_jsonl_or_tsv(more)
            if not parsed:
                parsed = parse_report_blocks(more)
            rows += parsed

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            if r.get("name") and r.get("operation"):
                f.write(json.dumps({"operation":r["operation"], "name":r["name"]})+"\n")
    print(f"Saved {sum(1 for _ in open(out))} kernels → {out}")

if __name__=="__main__":
    main()