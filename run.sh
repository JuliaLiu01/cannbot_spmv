set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="spmv_custom"
SKIP_BUILD=0
SHAPE="small"
INSTANCE=""
SCALE=1024

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --shape)
            SHAPE="$2"
            shift 2
            ;;
        --instance)
            INSTANCE="$2"
            shift 2
            ;;
        --scale)
            SCALE="$2"
            shift 2
            ;;
        *)
            echo "ERROR: unknown argument $1" >&2
            exit 1
            ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] 设置 CANN 环境 ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH 未设置，请先配置 CANN 环境"
if [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh 执行失败"
elif [ -f "$(dirname "${ASCEND_HOME_PATH}")/set_env.sh" ]; then
    source "$(dirname "${ASCEND_HOME_PATH}")/set_env.sh" || die "set_env.sh 执行失败"
else
    die "未找到 set_env.sh，请检查 ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
fi

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build 指定但 build/${OP_NAME} 不存在，请先完整编译"
    echo "=== [2/4] 跳过编译（复用已有产物）==="
else
    echo "=== [2/4] 编译 ==="
    mkdir -p build
    cd build
    cmake .. || die "cmake 配置失败"
    make -j4 || die "make 编译失败"
    cd ..
fi

echo "=== [3/4] 生成测试数据 ==="
cd build
if [ -n "${INSTANCE}" ]; then
    python3 ../scripts/gen_data.py --instance "${INSTANCE}" --scale "${SCALE}" || die "gen_data.py 执行失败"
else
    python3 ../scripts/gen_data.py --shape "${SHAPE}" || die "gen_data.py 执行失败"
fi

CONFIG=$(python3 - <<'PY' "${SHAPE}" "${INSTANCE}" "${SCALE}"
import csv
import json
import pathlib
import sys

shape, instance, scale = sys.argv[1], sys.argv[2], int(sys.argv[3])
shapes = {
    "tiny": {"rows": 8, "cols": 16, "density": 0.125},
    "small": {"rows": 256, "cols": 512, "density": 0.01},
    "medium": {"rows": 4096, "cols": 8192, "density": 0.001},
    "large": {"rows": 32768, "cols": 65536, "density": 0.0001},
}
if instance:
    csv_path = pathlib.Path('/root/liujunyan/analysis/cannbot_test/spmv_kernel/待测试的A矩阵基本信息.csv')
    with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        lines = [line for line in f if line.strip()]
    reader = csv.DictReader(lines, delimiter=';')
    for row in reader:
        if row['Instance'] == instance:
            cfg = {
                'rows': max(1, int(row['Number of Rows']) // scale),
                'cols': max(1, int(row['Number of Columns']) // scale),
                'density': float(row['Density (%)']) / 100.0,
            }
            print(json.dumps(cfg))
            break
    else:
        raise SystemExit(f'instance not found: {instance}')
else:
    print(json.dumps(shapes[shape]))
PY
)
ROWS=$(python3 - <<'PY' "${CONFIG}"
import json
import sys
print(json.loads(sys.argv[1])["rows"])
PY
)
COLS=$(python3 - <<'PY' "${CONFIG}"
import json
import sys
print(json.loads(sys.argv[1])["cols"])
PY
)
NNZ=$(python3 - <<'PY'
import numpy as np
print(np.fromfile('input/values.bin', dtype=np.float32).size)
PY
)

echo "=== [4/4] 运行 Kernel ==="
rm -f output/output.bin
"./${OP_NAME}" "${ROWS}" "${COLS}" "${NNZ}" || die "Kernel 运行失败"
[ -f output/output.bin ] || die "Kernel 运行后 output.bin 不存在"

echo "=== 精度验证 ==="
python3 ../scripts/verify_result.py output/output.bin output/golden.bin || die "精度验证失败"

echo "=== 完成 ==="
