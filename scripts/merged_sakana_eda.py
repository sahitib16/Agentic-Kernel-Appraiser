import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import shorten

# === Load merged dataset ===
m = pd.read_parquet("data/derived/final_data_all/merged_with_sakana.parquet")
print(f"Rows: {len(m):,}, Columns: {len(m.columns)}")

# === High-level stats ===
print("\nBasic label stats:")
print(m["correct"].value_counts(normalize=True).rename("fraction"))

print("\nCutlass vs Sakana split:")
print(m[["is_cutlass","is_sakana"]].sum())

print("\nNon-null code coverage:")
print(m[["code_text","ptx_code","cuda_code"]].notna().sum())

# === Plot: source composition ===
src_counts = pd.Series({
    "CUTLASS": m["is_cutlass"].sum(),
    "SAKANA": m["is_sakana"].sum(),
    "Both": ((m["is_cutlass"] & m["is_sakana"]).sum())
})
src_counts.plot(kind="bar", color="steelblue", figsize=(5,3), title="Dataset Source Composition")
plt.ylabel("Row count")
plt.tight_layout()
plt.show()

# === Plot: operation distribution ===
plt.figure(figsize=(8,4))
sns.countplot(y=m["op_kind"], order=m["op_kind"].value_counts().index, palette="viridis")
plt.title("Operation kind distribution")
plt.xlabel("Count")
plt.ylabel("Operation Kind")
plt.tight_layout()
plt.show()

# === Correctness by operation kind ===
corr_by_op = m.groupby("op_kind")["correct"].mean().sort_values(ascending=False)
plt.figure(figsize=(6,4))
sns.barplot(x=corr_by_op.values, y=corr_by_op.index, palette="crest")
plt.title("Fraction correct by operation kind")
plt.xlabel("Fraction correct")
plt.tight_layout()
plt.show()

# === Code length vs correctness ===
m["code_len"] = m["code_text"].str.len()
sns.histplot(data=m, x="code_len", hue="correct", bins=50, element="step", stat="percent")
plt.title("Distribution of code length by correctness")
plt.xlabel("Code length (characters)")
plt.tight_layout()
plt.show()

# === PTX coverage per op_kind ===
ptx_cov = m.groupby("op_kind")["ptx_code"].apply(lambda s: s.notna().mean()*100).round(1)
print("\nPTX coverage by op_kind (%):\n", ptx_cov)