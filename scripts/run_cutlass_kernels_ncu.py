#!/usr/bin/env python3
"""
Run Nsight Compute (ncu) over a JSONL kernel catalog and save artifacts.

Features:
- Accepts catalog rows with either:
  * "cmd" (full shell string), or
  * "exec" + "args" (argv style), or
  * no command -> synthesizes from row using:
      - --cmd-template (Python .format over row), or
      - --base-exec + auto --key=value flags (with --keymap, --const, --flag-style).
- Writes one directory per run: ncu.csv, stdout.txt, stderr.txt, meta.json.
- Puts everything under --outdir/<timestamp>/ (default: data/ncu/final/<ts>).
- Optional: treat specific return codes as success (--ok-codes 0 1).
- Optional: preview first N synthesized commands (--print-first N) and/or --dry-run.

Typical CUTLASS example:
  python scripts/run_cutlass_kernels_ncu.py data/ncu/kernel_catalog.jsonl \
    --base-exec "$CUTLASS_PROFILER" \
    --keymap op=operation M=m N=n K=k A=element-a B=element-b C=element-c \
    --const "--verification=none" "--iters=1" \
    --ncu ncu --preset none --timeout-sec 900 \
    --ok-codes 0 1 \
    --outdir data/ncu/final \
    --print-first 5 --dry-run

Then drop --dry-run once the printed commands look right.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

DEFAULT_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__inst_executed.sum",
    "sm__sass_thread_inst_executed_op_fma_pred_on.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "l1tex__t_bytes.sum",
    "lts__t_bytes.sum",
    "dram__bytes.sum",
    "gpu__time_duration.sum",
]

# Keys we do not convert into flags when auto-synthesizing
META_KEYS = {
    "id", "workdir", "env", "flags", "cmd", "exec", "args",
    "notes", "arch", "sm", "tag"
}

def parse_keymap(pairs):
    """Parse key renames like ['op=operation','A=element-a'] into dict."""
    mapping = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--keymap entry must be key=flagname, got: {p}")
        k, v = p.split("=", 1)
        mapping[k.strip()] = v.strip()
    return mapping

def parse_const(consts):
    """Parse constant flags that should always be appended."""
    out = []
    for c in consts or []:
        out.extend(shlex.split(c))
    return out

def parse_ok_codes(codes):
    """Return codes to treat as success."""
    return {int(x) for x in codes or [0]}

def parse_args():
    ap = argparse.ArgumentParser(description="Run Nsight Compute on CUTLASS kernel catalog.")
    ap.add_argument("catalog", type=str, help="Path to kernel_catalog.jsonl")
    ap.add_argument("--ncu", type=str, default="ncu", help="Path to Nsight Compute CLI")
    ap.add_argument("--preset", type=str, default="none",
                    help="ncu preset (speedOfLight,guided,full,none). If not 'none', --metrics is ignored.")
    ap.add_argument("--ncu-extra", type=str, nargs="*", default=[],
                help="Extra args appended to ncu (e.g. --ncu-extra --kernel-name-base cutlass_.* --launch-count 1)")
    ap.add_argument("--metrics", type=str, nargs="*", default=DEFAULT_METRICS,
                    help="Metrics list (ignored if --preset != none).")
    ap.add_argument("--target-processes", default="all", choices=["all", "application-only"],
                    help="ncu --target-processes (default: all)")
    ap.add_argument("--timeout-sec", type=int, default=600, help="Per-run wall clock timeout")
    ap.add_argument("--limit", type=int, default=0, help="Max rows to run (0 = all)")
    ap.add_argument("--outdir", type=str, default="data/ncu/final", help="Base output dir (default: data/ncu/final)")
    ap.add_argument("--dry-run", action="store_true", help="Print/record commands, do not execute")
    ap.add_argument("--print-first", type=int, default=0, help="Print synthesized app command for first N rows")

    # Command synthesis controls
    ap.add_argument("--base-exec", type=str, default=None,
                    help="If row lacks cmd/exec, run this binary and build flags from row/flags dict.")
    ap.add_argument("--cmd-template", type=str, default=None,
                    help="Python .format template for a full shell command. "
                         "Example: '{base_exec} --operation={operation} --m={M} --n={N} --k={K} "
                         "--element-a={A} --element-b={B} --element-c={C}'")
    ap.add_argument("--keymap", type=str, nargs="*", default=[],
                    help="Rename row keys to flag names, e.g. op=operation A=element-a B=element-b C=element-c")
    ap.add_argument("--const", type=str, nargs="*", default=[],
                    help="Constant flags to always append, e.g. '--verification=none' '--iters=1'")
    ap.add_argument("--flag-style", choices=["eq", "space"], default="eq",
                    help="Emit flags as '--k=v' (eq) or '--k v' (space)")
    ap.add_argument("--skip-keys", type=str, nargs="*", default=[],
                    help="Extra row keys to skip when auto-building flags")

    # Nsight return codes to accept as success
    ap.add_argument("--ok-codes", type=str, nargs="*", default=["0"],
                    help="Return codes to treat as success (e.g. 0 1)")
    return ap.parse_args()

def stringify_flag_value(v):
    if isinstance(v, bool):
        return "1" if v else None
    return None if v is None else str(v)

def as_flag(flag_name, sval, style):
    if sval is None:
        return None
    if style == "eq":
        return f"--{flag_name}={sval}"
    else:
        # space style -> two argv tokens later
        return f"--{flag_name} {shlex.quote(sval)}"

def row_to_flags(entry, skip_extra, keymap, style):
    """Convert a catalog row to flags, preferring entry['flags'] if provided."""
    if isinstance(entry.get("flags"), dict):
        items = entry["flags"].items()
    else:
        items = ((k, v) for k, v in entry.items()
                 if k not in META_KEYS and k not in skip_extra)

    out = []
    for k, v in items:
        sval = stringify_flag_value(v)
        if sval is None:
            continue
        flag_name = keymap.get(k, k).replace("_", "-")
        flag = as_flag(flag_name, sval, style)
        if flag:
            if style == "space":
                parts = flag.split(" ", 1)
                out.append(parts[0]); out.append(parts[1])
            else:
                out.append(flag)
    return out

def build_app_argv(entry, base_exec, cmd_template, keymap, const_flags, style, skip_extra):
    """Decide how to invoke the underlying app for a catalog row."""
    if entry.get("cmd"):
        return ["bash", "-lc", entry["cmd"]], entry["cmd"]

    if entry.get("exec"):
        exe = entry["exec"]
        args = [str(a) for a in entry.get("args", [])]
        display = " ".join([shlex.quote(exe)] + [shlex.quote(a) for a in args])
        return [exe] + args, display

    # Template first (most explicit)
    if cmd_template:
        ctx = dict(entry)
        if base_exec:
            ctx.setdefault("base_exec", base_exec)
        try:
            rendered = cmd_template.format(**ctx)
        except KeyError as e:
            raise ValueError(f"cmd-template missing key from row: {e}")
        return ["bash", "-lc", rendered], rendered

    # Otherwise auto-build flags from row / flags{} dict
    if not base_exec:
        raise ValueError("Row has no cmd/exec and no --base-exec/--cmd-template provided.")
    auto_flags = row_to_flags(entry, set(skip_extra), keymap, style)
    if not auto_flags:
        raise ValueError("Row produced no flags; provide --cmd-template or --keymap/--const to form a valid command.")
    argv = [base_exec] + auto_flags + const_flags
    display = " ".join([shlex.quote(base_exec)] + [shlex.quote(x) for x in auto_flags + const_flags])
    return argv, display

def build_ncu_cmd(app_argv, ncu_path, preset, metrics, target_processes, ncu_extra):
    ncu_argv = [ncu_path, "--csv", "--page", "raw", "--target-processes", target_processes]
    if preset and preset.lower() != "none":
        ncu_argv += ["--set", preset]
    else:
        if metrics:
            ncu_argv += ["--metrics", ",".join(metrics)]  # single comma-separated flag
    if ncu_extra:
        ncu_argv += ncu_extra
    full = ncu_argv + ["--"] + app_argv
    display = "{} -- {}".format(" ".join(map(shlex.quote, ncu_argv)), " ".join(map(shlex.quote, app_argv)))
    return full, display

def run_once(entry, outdir: Path, args, ok_codes):
    run_id = entry.get("id") or str(uuid.uuid4())
    run_dir = outdir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": run_id,
        "ts_start": time.time(),
        "catalog_entry": entry,
        "ncu": args.ncu,
        "preset": args.preset,
        "metrics": (None if args.preset.lower() != "none" else args.metrics),
        "target_processes": args.target_processes,
        "timeout_sec": args.timeout_sec,
        "status": "started",
        "return_code": None,
        "duration_sec": None,
        "ncu_csv": str(run_dir / "ncu.csv"),
        "stdout": str(run_dir / "stdout.txt"),
        "stderr": str(run_dir / "stderr.txt"),
    }

    workdir = entry.get("workdir")
    env = os.environ.copy()
    for k, v in (entry.get("env") or {}).items():
        env[str(k)] = str(v)
    try:
        safe_tmp = Path("data/ncu/tmp_ncu") / os.environ.get("USER", "ncu")
        safe_tmp.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(safe_tmp)
    except Exception as e:
        print(f"[WARN] Could not create local TMPDIR for ncu: {e}", file=sys.stderr)
    try:
        keymap = parse_keymap(args.keymap)
        const_flags = parse_const(args.const)
        app_argv, app_display = build_app_argv(entry, args.base_exec, args.cmd_template,
                                               keymap, const_flags, args.flag_style, args.skip_keys)
        full_cmd, printable = build_ncu_cmd(app_argv, args.ncu, args.preset, args.metrics,
                                    args.target_processes, args.ncu_extra)

        meta["app_display"] = app_display
        meta["command_printable"] = printable
        meta["workdir"] = workdir

        if args.dry_run:
            meta["status"] = "dry_run"
            Path(meta["stdout"]).write_text("")
            Path(meta["stderr"]).write_text("")
            Path(meta["ncu_csv"]).write_text("")
            (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            return meta

        with (run_dir / "stdout.txt").open("wb") as f_out, (run_dir / "stderr.txt").open("wb") as f_err:
            start = time.time()
            proc = subprocess.Popen(full_cmd, cwd=workdir, env=env, stdout=subprocess.PIPE, stderr=f_err)
            try:
                stdout_bytes, _ = proc.communicate(timeout=args.timeout_sec)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout_bytes, _ = proc.communicate()
                meta["status"] = "timeout"
                meta["duration_sec"] = time.time() - start
                f_out.write(stdout_bytes or b"")
                (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
                return meta

            f_out.write(stdout_bytes or b"")
            (run_dir / "ncu.csv").write_bytes(stdout_bytes or b"")
            meta["return_code"] = int(proc.returncode)
            meta["duration_sec"] = time.time() - start
            meta["status"] = "ok" if proc.returncode in ok_codes else "ncu_error"

    except Exception as e:
        meta["status"] = f"exception: {type(e).__name__}: {e}"

    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta

def pick_metrics(ncu_path: str) -> list[str]:
    """
    Return a safe metrics list:
      - Counters that exist on Ada (L40S)
      - Add kernel time if available on this ncu build
    """
    desired = [
        "dram__bytes.sum",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "gpc__cycles_elapsed.sum",
        # Optional duration metric(s) depending on ncu build
        "gpu__time_duration.sum",
        "gpu__time_duration",   # some builds expose the plain name
    ]
    # Probe availability (respect TMPDIR env you already set)
    try:
        out = subprocess.run(
            [ncu_path, "--query-metrics"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True
        ).stdout
    except Exception:
        out = ""
    avail = set()
    for line in out.splitlines():
        # metric name is the first token up to whitespace
        if not line or line.startswith("Device "): continue
        name = line.split()[0]
        avail.add(name)
        # also add rollups like .sum/.avg manually if printed without them
        if "." not in name:
            avail.add(name+".sum")
            avail.add(name+".avg")
    picked = []
    for m in desired:
        base = m.split(".")[0]  # match either bare or rolled-up
        if m in avail or base in avail:
            picked.append(m)
    # Always at least the 4 counters:
    base_counters = [
        "dram__bytes.sum","dram__bytes_read.sum","dram__bytes_write.sum","gpc__cycles_elapsed.sum"
    ]
    for m in base_counters:
        if m not in picked:
            picked.append(m)
    return picked

def main():
    args = parse_args()
    if args.ncu_extra:
        args.ncu_extra = shlex.split(" ".join(args.ncu_extra))
    if args.preset.lower() == "none":
        args.metrics = pick_metrics(args.ncu)

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        print(f"ERROR: Catalog file not found: {catalog_path}", file=sys.stderr)
        sys.exit(2)

    # Output dir -> timestamped subdir under requested base
    base = Path(args.outdir)
    base.mkdir(parents=True, exist_ok=True)
    outdir = base / f"ncu_runs_{int(time.time())}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing results under: {outdir}")

    # ncu presence check
    try:
        subprocess.run([args.ncu, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError:
        print(f"ERROR: Could not find ncu at '{args.ncu}'. Provide --ncu /full/path/to/ncu", file=sys.stderr)
        sys.exit(3)

    ok_codes = parse_ok_codes(args.ok_codes)

    manifest = outdir / "runs.jsonl"
    total = 0
    ok = 0
    with catalog_path.open("r") as f_in, manifest.open("a") as f_manifest:
        for line_idx, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping bad JSONL line {line_idx}: {e}")
                continue

            total += 1
            if args.limit and total > args.limit:
                break

            meta = run_once(entry, outdir, args, ok_codes)
            f_manifest.write(json.dumps(meta) + "\n")

            if args.print_first and total <= args.print_first:
                print(f"[CMD] {meta.get('app_display','')}")

            status = meta.get("status")
            rc = meta.get("return_code")
            if status == "ok":
                ok += 1
                dur = meta.get("duration_sec")
                print(f"[OK ] {meta['run_id']}: rc={rc} ({dur:.2f}s)")
            else:
                print(f"[FAIL] {meta['run_id']}: {status}; rc={rc}")

    print(f"[DONE] {ok}/{total} runs succeeded. Manifest: {manifest}")
    print(f"[DIR ] Artifacts: {outdir}")

if __name__ == "__main__":
    main()