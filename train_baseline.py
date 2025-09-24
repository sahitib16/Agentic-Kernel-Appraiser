import os
os.environ["MPLBACKEND"] = "Agg"  

import pandas as pd, numpy as np, xgboost as xgb, shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score
from scipy.stats import spearmanr
from pathlib import Path
import matplotlib.pyplot as plt

def target_encode_train_test(train_col, test_col, y_train, k=20):
    prior = float(y_train.mean())
    counts = train_col.value_counts()
    means  = train_col.groupby(train_col).apply(lambda s: y_train.loc[s.index].mean())
    smooth = counts / (counts + k)
    enc_map = smooth * means + (1 - smooth) * prior
    return train_col.map(enc_map).fillna(prior), test_col.map(enc_map).fillna(prior)

DATA = Path("data/features_v0.parquet")
OUT  = Path("data")
OUT.mkdir(exist_ok=True, parents=True)

def one_hot(df, cols):
    return pd.get_dummies(df, columns=cols, dummy_na=True)

def build_X(df):
    base = ["uses_shared","uses_vec4","uses_unroll",
            "code_len","kernel_len","dim0","dim1","dim2",
            "pytorch_native_runtime",
            "count_sync","count_for","count_if"]
    base = [c for c in base if c in df.columns]
    # only level_id is one-hot; op_name is handled via target encoding per split
    cats = ["level_id"]
    X = df[base + cats].copy()
    X = one_hot(X, cats)
    return X

def save_pr_curve(y_true, y_prob, path):
    p, r, t = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(4,3))
    plt.plot(r, p)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR curve (AP={ap:.3f})")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()

def best_f1_threshold(y_true, y_prob):
    p, r, t = precision_recall_curve(y_true, y_prob)
    f1s = 2*p*r/(p+r+1e-12)
    idx = np.nanargmax(f1s)
    thr = t[idx] if idx < len(t) else 0.5
    return float(thr), float(np.nanmax(f1s))

def correctness_head(df):
    X = build_X(df)
    y = df["label_correct"].astype(int)
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    # add op_name_te (mean correctness per op_name computed on training fold)
    op_tr, op_te = target_encode_train_test(df.loc[Xtr.index, "op_name"],
                                            df.loc[Xte.index, "op_name"],
                                            ytr, k=20)
    Xtr["op_name_te"] = op_tr.values
    Xte["op_name_te"] = op_te.values

    clf = xgb.XGBClassifier(
        n_estimators=400,max_depth=6,learning_rate=0.1,
        subsample=0.8,colsample_bytree=0.8,eval_metric="logloss"
    )
    clf.fit(Xtr,ytr)
    y_prob = clf.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, y_prob)
    ap  = average_precision_score(yte, y_prob)
    thr, best_f1 = best_f1_threshold(yte, y_prob)

    # confusion table best-F1 and at 0.5
    for tag, th in [("bestF1",thr),("thr0.5",0.5)]:
        y_hat = (y_prob >= th).astype(int)
        tp = int(((y_hat==1)&(yte==1)).sum()); fp = int(((y_hat==1)&(yte==0)).sum())
        fn = int(((y_hat==0)&(yte==1)).sum()); tn = int(((y_hat==0)&(yte==0)).sum())
        print(f"[correctness] AUC={auc:.3f}  AP={ap:.3f}  {tag}: thr={th:.3f} F1={f1_score(yte,y_hat):.3f} "
              f"tp={tp} fp={fp} fn={fn} tn={tn}")

    # PR curve
    save_pr_curve(yte, y_prob, OUT/"pr_correctness.png")

    # SHAP summary + dependence plots
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xte)
    shap.summary_plot(sv, Xte, show=False, max_display=20)
    plt.tight_layout(); plt.savefig(OUT/"shap_correctness.png", dpi=160); plt.close()
    print("Saved:", OUT/"shap_correctness.png", OUT/"pr_correctness.png")

    # dependence plots for top 3 features
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:3]
    for i in top_idx:
        fname = Xte.columns[i]
        try:
            shap.dependence_plot(i, sv, Xte, show=False)
            plt.tight_layout(); plt.savefig(OUT/f"shap_correctness_dep_{fname}.png", dpi=160); plt.close()
        except Exception:
            pass
    return clf

def speedup_head(df):
    dfc = df[df["label_correct"]==1].copy()
    y = np.log1p(dfc["label_speedup"].clip(lower=0))
    dfc = df[df["label_correct"]==1].copy()
    y = np.log1p(dfc["label_speedup"].clip(lower=0))
    X = build_X(dfc)
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)

    op_tr, op_te = target_encode_train_test(dfc.loc[Xtr.index, "op_name"],
                                            dfc.loc[Xte.index, "op_name"],
                                            ytr, k=20)
    Xtr["op_name_te"] = op_tr.values
    Xte["op_name_te"] = op_te.values

    reg = xgb.XGBRegressor(
        n_estimators=600,max_depth=7,learning_rate=0.06,
        subsample=0.8,colsample_bytree=0.8
    )
    reg.fit(Xtr,ytr)
    y_pred = reg.predict(Xte)
    sp = spearmanr(yte, y_pred).correlation
    print("Speedup (log1p) Spearman:", round(float(sp),4))

    explainer = shap.TreeExplainer(reg)
    sv = explainer.shap_values(Xte)
    shap.summary_plot(sv, Xte, show=False, max_display=20)
    plt.tight_layout(); plt.savefig(OUT/"shap_speedup.png", dpi=160); plt.close()
    print("Saved:", OUT/"shap_speedup.png")

    # dependence for top 3
    importances = reg.feature_importances_
    top_idx = np.argsort(importances)[::-1][:3]
    for i in top_idx:
        fname = Xte.columns[i]
        try:
            shap.dependence_plot(i, sv, Xte, show=False)
            plt.tight_layout(); plt.savefig(OUT/f"shap_speedup_dep_{fname}.png", dpi=160); plt.close()
        except Exception:
            pass
    return reg

if __name__=="__main__":
    df = pd.read_parquet(DATA)
    print("Rows:", len(df))
    clf = correctness_head(df)
    reg = speedup_head(df)