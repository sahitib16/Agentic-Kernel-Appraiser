import os
import argparse
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def save(fig, out_base):
    # save interactive HTML + static PNG (needs kaleido)
    fig.write_html(out_base + ".html", include_plotlyjs="cdn")
    try:
        fig.write_image(out_base + ".png", scale=2)
    except Exception as e:
        print("[warn] PNG export skipped:", e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="merged_with_sakana.parquet")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    m = pd.read_parquet(args.inp)
    print(f"Rows: {len(m):,}, Cols: {len(m.columns)}")

    # basic prints
    print("\nLabel fraction (correct):")
    print(m["correct"].value_counts(normalize=True).rename("fraction"))
    print("\nCode coverage:")
    print(m[["code_text","ptx_code","cuda_code"]].notna().sum())

    # 1) Source composition
    src_counts = pd.DataFrame({
        "source": ["CUTLASS","SAKANA","Both"],
        "count": [
            int(m["is_cutlass"].sum()),
            int(m["is_sakana"].sum()),
            int(((m["is_cutlass"]==1) & (m["is_sakana"]==1)).sum()),
        ]
    })
    fig1 = px.bar(src_counts, x="source", y="count", title="Dataset Source Composition",
                  text="count")
    fig1.update_traces(textposition="outside")
    save(fig1, os.path.join(args.outdir, "source_composition"))

    # 2) Operation kind distribution
    if "op_kind" in m.columns:
        op_counts = m["op_kind"].value_counts().reset_index()
        op_counts.columns = ["op_kind","count"]
        fig2 = px.bar(op_counts, x="count", y="op_kind", orientation="h",
                      title="Operation Kind Distribution")
        save(fig2, os.path.join(args.outdir, "operation_distribution"))

        # 3) Fraction correct by op_kind
        corr_by_op = m.groupby("op_kind")["correct"].mean().reset_index()
        corr_by_op.sort_values("correct", ascending=False, inplace=True)
        fig3 = px.bar(corr_by_op, x="correct", y="op_kind", orientation="h",
                      title="Fraction Correct by op_kind",
                      labels={"correct":"fraction correct"})
        save(fig3, os.path.join(args.outdir, "correctness_by_opkind"))

    # 4) Code length histogram (by correctness)
    m["code_len"] = m["code_text"].str.len()
    fig4 = px.histogram(m, x="code_len", color=m["correct"].map({1:"correct",0:"incorrect"}),
                        nbins=50, barmode="overlay",
                        title="Code Length Distribution by Correctness")
    fig4.update_traces(opacity=0.65)
    fig4.update_layout(xaxis_title="Code length (characters)")
    save(fig4, os.path.join(args.outdir, "code_length_histogram"))

    # 5) PTX coverage per op_kind (table + bar if available)
    if "op_kind" in m.columns:
        ptx_cov = m.groupby("op_kind")["ptx_code"].apply(lambda s: s.notna().mean()*100).round(1)
        ptx_cov.to_csv(os.path.join(args.outdir, "ptx_coverage_by_opkind.csv"))
        ptx_df = ptx_cov.reset_index(name="ptx_coverage_pct")
        fig5 = px.bar(ptx_df.sort_values("ptx_coverage_pct"),
                      x="ptx_coverage_pct", y="op_kind", orientation="h",
                      title="PTX Coverage by op_kind (%)")
        save(fig5, os.path.join(args.outdir, "ptx_coverage_by_opkind"))

    print(f"\n✅ Saved Plotly outputs to: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()