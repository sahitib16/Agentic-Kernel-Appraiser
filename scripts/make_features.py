import pandas as pd
from ksn.features import simple_features

def main():
    # Load the subset 
    df = pd.read_parquet("data/sakana_light.parquet")
    print("Loaded sakana_light.parquet with", len(df), "rows")

    # generate features and their labels
    feats = simple_features(df)
    print(feats.columns)
    feats.to_parquet("data/features_v0.parquet")

    print("Feature table shape:", feats.shape)
    print("First few rows:")
    print(feats.head(5))
    print(feats.columns)

    print("\nCorrect label distribution:")
    print(feats['label_correct'].value_counts(dropna=False))

    print("\nSpeedup stats:")
    print(feats['label_speedup'].describe())

if __name__ == "__main__":
    main()

f = pd.read_parquet("data/features_v0.parquet")

# correct vs incorrect speedups
print(f.groupby('label_correct')['label_speedup'].describe())

print(f.groupby('label_correct')[['uses_shared','uses_unroll','uses_vec4']].mean())

# speedup by op
print(f.groupby('op_name')['label_speedup'].median().sort_values(ascending=False).head(10))

print("Shared memory usage and median speedup:")
print(f.groupby('uses_shared')['label_speedup'].median())

print("Vectorization usage and median speedup:")
print(f.groupby('uses_vec4')['label_speedup'].median())