import argparse
import subprocess
from pathlib import Path

def sh(cmd: str):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g., data/ncu/all/20251014-014945")
    ap.add_argument("--outdir", required=True, help="e.g., data/derived/20251014-014945_all")
    ap.add_argument("--include-subfolders", default="ncu_conv,ncu_gemm")
    ap.add_argument("--normalize", action="store_true", help="Normalize/patch metrics in-place after manifest")
    ap.add_argument("--analyze", action="store_true", help="Run analyze_features after features are built")
    ap.add_argument("--explore", action="store_true", help="Run explore_merged after features are built")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # 1) Build combined manifest + metrics
    sh(
        f"python -m scripts.build_manifest "
        f"--root {args.root} "
        f"--include-subfolders {args.include_subfolders} "
        f"--outdir {args.outdir} "
        f"--write-clean-csv"
    )

    manifest_p = str(Path(args.outdir) / "manifest.parquet")
    metrics_p  = str(Path(args.outdir) / "ncu_metrics.parquet")

    # 2) Optional: normalize + patch kernel names from manifest (robust if earlier steps changed)
    if args.normalize:
        sh(
            f"python -m scripts.normalize_ncu_metrics "
            f"--in {metrics_p} "
            f"--manifest {manifest_p} "
            f"--out {metrics_p}"
        )

    # 3) Build features and merged table
    sh(
        f"python -m scripts.make_features "
        f"--manifest {manifest_p} "
        f"--metrics {metrics_p} "
        f"--outdir {args.outdir}"
    )

    merged_p = str(Path(args.outdir) / "merged_all.parquet")

    # 4) Optional: analyze and/or explore
    if args.analyze:
        sh(
            f"python -m scripts.analyze_features "
            f"--merged {merged_p} "
            f"--outdir {args.outdir} "
            f"--drop-features operation "
            f"--onehot conv_role,layout,dtype,math_mode,tile_shape_bucket "
            f"--target gpu__time_duration.sum --log-target"
        )

    if args.explore:
        sh(
            f"python -m scripts.explore_merged "
            f"--merged {merged_p} "
            f"--outdir {args.outdir}"
        )

if __name__ == "__main__":
    main()