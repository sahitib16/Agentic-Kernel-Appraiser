# scripts/analyze_failure_modes.py
import os
import re
import json
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)


def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)


def tokenize_code(s):
    if not isinstance(s, str):
        return []
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_\.]*", s)
    return [t for t in toks if not t.isdigit()][:5000]


def describe_split(df, name):
    print(f"[{name}] rows={len(df):,}, correct=1 frac={df['correct'].mean():.3f}")
    lens = df["code_text"].fillna("").str.len()
    print(f"  code length char p50={lens.median():.0f}, p90={lens.quantile(0.9):.0f}, max={lens.max():.0f}")


def plot_confusion(cm, labels, out_path, title="Confusion Matrix"):
    plt.figure(figsize=(4, 3))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                     xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_hist_by_label(series, labels, out_path, title, xlabel):
    plt.figure(figsize=(6, 3.5))
    sns.histplot(data=series.to_frame("x").join(labels.rename("correct")),
                 x="x", hue="correct", bins=50, stat="percent", element="step")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to value_model_xgb.joblib")
    ap.add_argument("--train", required=True, help="train_split.parquet")
    ap.add_argument("--val",   required=True, help="val_split.parquet")
    ap.add_argument("--test",  required=True, help="test_split.parquet (code-only)")
    ap.add_argument("--outdir", required=True, help="output analysis dir")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # Load artifacts
    print("[load] model & data")
    pipe = joblib.load(args.model)
    train = pd.read_parquet(args.train)
    val   = pd.read_parquet(args.val)
    test  = pd.read_parquet(args.test)

    # Basic descriptions
    for df, name in [(train, "train"), (val, "val"), (test, "test(code-only)")]:
        describe_split(df, name)

    # Columns available in each split
    text_col = "code_text"
    # only use fields present in each split
    fe_cols_train = [c for c in [text_col, "op_kind", "dtype", "layout"] if c in train.columns]
    fe_cols_val   = [c for c in [text_col, "op_kind", "dtype", "layout"] if c in val.columns]
    fe_cols_test  = [c for c in [text_col, "op_kind", "dtype", "layout"] if c in test.columns]

    # Predictions
    def predict(df, fe_cols):
        proba = pipe.predict_proba(df[fe_cols])[:, 1]
        preds = (proba >= 0.5).astype(int)
        return preds, proba

    print("[predict] val")
    yv = val["correct"].astype(int).values
    yv_pred, yv_proba = predict(val, fe_cols_val)

    print("[predict] test (code-only)")
    yt = test["correct"].astype(int).values
    yt_pred, yt_proba = predict(test, fe_cols_test)

    # Reports
    print("\n[VAL] classification report")
    val_report = classification_report(yv, yv_pred, digits=3, output_dict=True)
    print(classification_report(yv, yv_pred, digits=3))
    val_auc = roc_auc_score(yv, yv_proba)
    print("VAL AUC:", val_auc)

    print("\n[TEST] classification report (code-only)")
    test_report = classification_report(yt, yt_pred, digits=3, output_dict=True)
    print(classification_report(yt, yt_pred, digits=3))
    test_auc = roc_auc_score(yt, yt_proba)
    print("TEST AUC:", test_auc)

    # Save reports as JSON
    with open(os.path.join(args.outdir, "val_report.json"), "w") as f:
        json.dump({"report": val_report, "auc": float(val_auc)}, f, indent=2)
    with open(os.path.join(args.outdir, "test_report.json"), "w") as f:
        json.dump({"report": test_report, "auc": float(test_auc)}, f, indent=2)

    # Confusion Matrices
    cm_val  = confusion_matrix(yv, yv_pred, labels=[0,1])
    cm_test = confusion_matrix(yt, yt_pred, labels=[0,1])
    plot_confusion(cm_val,  ["0(incorrect)","1(correct)"],
                   os.path.join(args.outdir, "cm_val.png"), "Confusion (VAL)")
    plot_confusion(cm_test, ["0(incorrect)","1(correct)"],
                   os.path.join(args.outdir, "cm_test.png"), "Confusion (TEST code-only)")

    # ROC & PR curves (test)
    fpr, tpr, _ = roc_curve(yt, yt_proba)
    ap = average_precision_score(yt, yt_proba)
    prec, rec, thr = precision_recall_curve(yt, yt_proba)

    plt.figure(figsize=(4.2,3.6))
    plt.plot(fpr, tpr, label=f"AUC={test_auc:.3f}")
    plt.plot([0,1],[0,1], "k--", alpha=0.4)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC (TEST)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "roc_test.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(4.2,3.6))
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (TEST)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "pr_test.png"), dpi=200)
    plt.close()

    # Threshold sweep CSV (test)
    thr = thr.tolist()
    df_thr = pd.DataFrame({
        "threshold": [np.nan] + thr,  # precision_recall_curve returns len(thr)=len(prec)-1
        "precision": prec.tolist(),
        "recall": rec.tolist()
    })
    df_thr.to_csv(os.path.join(args.outdir, "threshold_sweep_test.csv"), index=False)

    # Per-op_kind performance (test) — if op_kind present in test
    if "op_kind" in test.columns:
        perop = []
        for op, grp in test.assign(pred=yt_pred, proba=yt_proba).groupby("op_kind"):
            acc = (grp["correct"].values == grp["pred"].values).mean()
            auc = roc_auc_score(grp["correct"].values, grp["proba"].values) if grp["correct"].nunique() > 1 else np.nan
            perop.append({"op_kind": op, "n": len(grp), "acc": acc, "auc": auc})
        df_perop = pd.DataFrame(perop).sort_values(["acc","n"], ascending=[False, False])
        df_perop.to_csv(os.path.join(args.outdir, "per_opkind_test.csv"), index=False)

        plt.figure(figsize=(6,4))
        sns.barplot(data=df_perop, x="acc", y="op_kind", palette="Blues_r")
        plt.title("Accuracy by op_kind (TEST)")
        plt.xlabel("Accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "per_opkind_accuracy_test.png"), dpi=200)
        plt.close()

    # False Positives / False Negatives sampling (test)
    # FP: predicted 1 but label 0; FN: predicted 0 but label 1
    preds_df = test[[text_col, "op_kind", "dtype", "layout", "math_mode", "conv_role", "correct"]].copy()
    preds_df["pred"] = yt_pred
    preds_df["proba"] = yt_proba
    preds_df["code_len"] = preds_df[text_col].str.len()

    fp = preds_df[(preds_df["pred"] == 1) & (preds_df["correct"] == 0)].copy()
    fn = preds_df[(preds_df["pred"] == 0) & (preds_df["correct"] == 1)].copy()

    # Save top-50 most confident mistakes for quick review
    fp.sort_values("proba", ascending=False).head(50).to_csv(os.path.join(args.outdir, "false_positives_top50.csv"), index=False)
    fn.sort_values("proba", ascending=True).head(50).to_csv(os.path.join(args.outdir, "false_negatives_top50.csv"), index=False)

    # Code length diagnostics on mistakes
    plot_hist_by_label(preds_df["code_len"], preds_df["correct"],
                       os.path.join(args.outdir, "code_length_by_correctness_test.png"),
                       "Code Length vs Correctness (TEST)", "Code length (chars)")

    # Simple token frequency view on mistakes (rough insight)
    def top_tokens(df, k=30):
        tokens = []
        for s in df[text_col].dropna().head(1000):  # cap to keep fast
            tokens.extend(tokenize_code(s))
        vc = pd.Series(tokens).value_counts().head(k)
        return vc

    top_fp = top_tokens(fp, 30)
    top_fn = top_tokens(fn, 30)
    top_fp.to_csv(os.path.join(args.outdir, "tokens_fp_top30.csv"))
    top_fn.to_csv(os.path.join(args.outdir, "tokens_fn_top30.csv"))

    print("\n[done] wrote analysis to:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()