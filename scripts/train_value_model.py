# scripts/train_value_model.py
import pandas as pd, numpy as np, os, argparse, joblib
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    train = pd.read_parquet(args.train)
    val   = pd.read_parquet(args.val)
    test  = pd.read_parquet(args.test)

    print(f"Train={len(train):,}, Val={len(val):,}, Test={len(test):,}")

    # ---------- Features ----------
    text_col = "code_text"
    cat_cols = [c for c in ["op_kind","dtype","layout"] if c in train.columns]
    target   = "correct"

    # hashing vectorizer for code
    text_vect = HashingVectorizer(
        n_features=2**16,  # adjust for memory
        ngram_range=(1,2),
        norm=None,
        alternate_sign=False
    )

    # one-hot for categorical
    cat_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    # build preprocessing pipeline
    preproc = ColumnTransformer([
        ("code", text_vect, text_col),
        ("cats", cat_enc, cat_cols)
    ])

    # XGBoost classifier
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=8
    )

    # Combine into full pipeline
    pipe = Pipeline([
        ("prep", preproc),
        ("clf", model)
    ])

    # ---------- Fit ----------
    X_train, y_train = train[[text_col]+cat_cols], train[target]
    X_val,   y_val   = val[[text_col]+cat_cols], val[target]
    X_test,  y_test  = test[[text_col]+cat_cols], test[target]

    print("\n[Training XGBoost classifier…]")
    pipe.fit(X_train, y_train)

    # ---------- Eval ----------
    def eval_split(X, y, name):
        preds = pipe.predict(X)
        proba = pipe.predict_proba(X)[:,1]
        print(f"\n{name} results:")
        print(classification_report(y, preds, digits=3))
        print("AUC:", roc_auc_score(y, proba))

    eval_split(X_val, y_val, "VALIDATION")
    eval_split(X_test, y_test, "TEST (code-only)")

    # ---------- Save ----------
    joblib.dump(pipe, os.path.join(args.outdir,"value_model_xgb.joblib"))
    print(f"\n✅ Model saved to {args.outdir}/value_model_xgb.joblib")

if __name__ == "__main__":
    main()