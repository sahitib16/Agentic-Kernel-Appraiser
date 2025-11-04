import argparse, pandas as pd, numpy as np
from pathlib import Path

def one_hot(df, cols):
    cols = [c for c in cols if c in df.columns]
    return pd.get_dummies(df, columns=cols, dummy_na=True) if cols else df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, help="merged_all.parquet")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--filter-operation", default=None, help="e.g., conv2d, gemm")
    ap.add_argument("--drop-features", default="", help="comma-separated list to drop, e.g., operation,conv_role")
    ap.add_argument("--onehot", default="conv_role,layout,dtype,math_mode,tile_shape_bucket",
                    help="comma-separated categoricals to one-hot")
    ap.add_argument("--target", default="gpu__time_duration.sum",
                    help="metric to predict (e.g., gpu__time_duration.sum or sm__throughput.avg.pct_of_peak_sustained_elapsed)")
    ap.add_argument("--log-target", action="store_true", help="use log1p(target)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.merged).copy()

    if args.filter_operation:
        df = df[df["operation"] == args.filter_operation].copy()

    # Target
    y = pd.to_numeric(df[args.target], errors="coerce")
    keep = y.notna()
    df = df.loc[keep].copy(); y = y.loc[keep]

    # Feature frame
    drop_cols = set([
        "run_dir","kernel_name","Kernel Name","ncu_kernel_symbol",
        # known metrics we don't want as X unless explicitly targeted
        "gpu__time_duration.sum","dram__bytes.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    ])
    # user-specified drops
    for c in [s.strip() for s in args.drop_features.split(",") if s.strip()]:
        drop_cols.add(c)

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # One-hot categoricals
    cats = [s.strip() for s in args.onehot.split(",") if s.strip()]
    X = one_hot(X, cats)

    # Numeric cleanup
    # Clean NaNs/Infs first
    X = X.replace([np.inf, -np.inf], np.nan)
    # Convert obvious numeric-like objects and avoid silent downcasting warning
    X = X.infer_objects(copy=False)
    # Drop any leftover object columns (e.g., free-form strings like tile_sig_raw)
    obj_cols = [c for c in X.columns if X[c].dtype == "object"]
    if obj_cols:
        X = X.drop(columns=obj_cols)
    # Fill remaining NaNs
    X = X.fillna(0)

    y_train = np.log1p(y) if args.log_target else y

    # Model (XGB if available; else RF)
    try:
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, subsample=0.9, colsample_bytree=0.9,
            learning_rate=0.05, random_state=42, n_jobs=8
        )
        algo = "xgboost"
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=600, max_depth=None, random_state=42, n_jobs=8
        )
        algo = "random_forest"

    model.fit(X, y_train)

    # Importance
    try:
        importances = model.feature_importances_
    except Exception:
        importances = getattr(model, "feature_importances_", None)
    imp = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values("importance", ascending=False)

    # Simple fit metric on train (quick sanity)
    pred = model.predict(X)
    if args.log_target:
        pred = np.expm1(pred)
    # R^2 (on train)
    r2 = 1 - np.sum((y - pred)**2) / np.sum((y - y.mean())**2)

    # Save
    imp_path = outdir / "feature_importance.csv"
    imp.to_csv(imp_path, index=False)

    print(f"[{algo}] target={args.target} log={args.log_target} filter_op={args.filter_operation} r2_train={r2:.3f}")
    print(imp.head(25).to_string(index=False))
    print("\nSaved:", imp_path)

if __name__ == "__main__":
    main()