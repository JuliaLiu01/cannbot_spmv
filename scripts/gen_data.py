import argparse
import csv
import os
from pathlib import Path
import numpy as np

SHAPES = {
    "tiny": {"rows": 8, "cols": 16, "density": 0.125},
    "small": {"rows": 256, "cols": 512, "density": 0.01},
    "medium": {"rows": 4096, "cols": 8192, "density": 0.001},
    "large": {"rows": 32768, "cols": 65536, "density": 0.0001},
}

INSTANCE_CSV = Path(__file__).resolve().parents[1] / "待测试的A矩阵基本信息.csv"


def load_instance(instance_name: str):
    with INSTANCE_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        lines = [line for line in f if line.strip()]
    reader = csv.DictReader(lines, delimiter=';')
    for row in reader:
        if row["Instance"] == instance_name:
            return {
                "rows": int(row["Number of Rows"]),
                "cols": int(row["Number of Columns"]),
                "density": float(row["Density (%)"]) / 100.0,
            }
    raise ValueError(f"Instance not found: {instance_name}")


def resolve_config(args):
    if args.instance:
        cfg = load_instance(args.instance)
        scale = max(1, args.scale)
        return {
            "rows": max(1, cfg["rows"] // scale),
            "cols": max(1, cfg["cols"] // scale),
            "density": cfg["density"],
            "label": f"instance:{args.instance}/scale:{scale}",
        }

    cfg = SHAPES[args.shape].copy()
    cfg["label"] = args.shape
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--shape", default="small", choices=SHAPES.keys())
parser.add_argument("--instance", default="")
parser.add_argument("--scale", type=int, default=1024)
args = parser.parse_args()

cfg = resolve_config(args)
rows = cfg["rows"]
cols = cfg["cols"]
density = cfg["density"]
label = cfg["label"]

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

rng = np.random.default_rng(20260422)
row_ptr = np.zeros(rows + 1, dtype=np.int32)
col_idx_parts = []
values_parts = []

target_per_row = max(1, int(cols * density))
for row in range(rows):
    nnz_row = min(cols, max(1, int(rng.poisson(target_per_row))))
    cols_this_row = np.sort(rng.choice(cols, size=nnz_row, replace=False).astype(np.int32))
    vals_this_row = rng.standard_normal(nnz_row, dtype=np.float32)
    col_idx_parts.append(cols_this_row)
    values_parts.append(vals_this_row)
    row_ptr[row + 1] = row_ptr[row] + nnz_row

if col_idx_parts:
    col_idx = np.concatenate(col_idx_parts).astype(np.int32, copy=False)
    values = np.concatenate(values_parts).astype(np.float32, copy=False)
else:
    col_idx = np.zeros((0,), dtype=np.int32)
    values = np.zeros((0,), dtype=np.float32)

x = rng.standard_normal(cols, dtype=np.float32)
golden = np.zeros(rows, dtype=np.float32)
for row in range(rows):
    start = row_ptr[row]
    end = row_ptr[row + 1]
    if start != end:
        golden[row] = np.dot(values[start:end], x[col_idx[start:end]])

row_ptr.tofile("input/row_ptr.bin")
col_idx.tofile("input/col_idx.bin")
values.tofile("input/values.bin")
x.tofile("input/x.bin")
golden.tofile("output/golden.bin")

print(f"Generated CSR data for shape={label}")
print(f"  rows={rows}, cols={cols}, density={density}, nnz={len(values)}")
print(f"  input/row_ptr.bin: {row_ptr.shape}, {row_ptr.dtype}")
print(f"  input/col_idx.bin: {col_idx.shape}, {col_idx.dtype}")
print(f"  input/values.bin: {values.shape}, {values.dtype}")
print(f"  input/x.bin: {x.shape}, {x.dtype}")
print(f"  output/golden.bin: {golden.shape}, {golden.dtype}")
