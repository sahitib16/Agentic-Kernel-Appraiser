
from datasets import load_dataset, concatenate_datasets
import pandas as pd

SPLITS = ["level_1", "level_2", "level_3"]
parts = [load_dataset("SakanaAI/AI-CUDA-Engineer-Archive", split=s) for s in SPLITS]
ds = concatenate_datasets(parts)

# make stuff lowercase to make easier
RENAME = {
    "Op_Name": "op_name",
    "Level_ID": "level_id",
    "Task_ID": "task_id",
    "Kernel_Name": "kernel_name",
    "CUDA_Runtime": "cuda_runtime",
    "PyTorch_Native_Runtime": "pytorch_native_runtime",
    "PyTorch_Compile_Runtime": "pytorch_compile_runtime",
    "CUDA_Speedup_Native": "cuda_speedup_native",
    "CUDA_Speedup_Compile": "cuda_speedup_compile",
    "CUDA_Code": "cuda_code",
    "PyTorch_Code_Module": "pytorch_code_module",
    "PyTorch_Code_Functional": "pytorch_code_functional",
    "Correct": "correct",
    "Max_Diff": "max_diff",
    "Error": "error",
    "NCU_Profile": "ncu_profile",
    "Torch_Profile": "torch_profile",
    "Clang_Tidy": "clang_tidy",
}

keep = list(RENAME.keys())
rows = []
for i in range(len(ds)):
    r = ds[i]
    rows.append({RENAME[k]: r.get(k, None) for k in keep})

df = pd.DataFrame(rows)

# cleaning
num = ["cuda_runtime","pytorch_native_runtime","pytorch_compile_runtime",
       "cuda_speedup_native","cuda_speedup_compile","max_diff"]
for c in num:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype("int8")

df["cuda_speedup_native"] = df["cuda_speedup_native"].clip(lower=0)
df["cuda_speedup_compile"] = df["cuda_speedup_compile"].clip(lower=0)


df.to_parquet("data/sakana_all.parquet")
df.sample(min(len(df), 20000), random_state=42).to_parquet("data/sakana_light.parquet")
print("saved parquet:", len(df))
print(df.columns)
failures = df[df["correct"] == 0]

print("\nTotal failures:", len(failures))

print("\nMost common error messages:")
print(failures["error"].value_counts().head(3))