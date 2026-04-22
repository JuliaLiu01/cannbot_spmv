set -euo pipefail

# 仅做：拉 CANN 环境 → 编译 → 运行可执行文件，对 kDatasetDir 下所有 .mps 做 SpMV 基准（见 main.cpp）。
# 数据/向量/golden 路径在 main.cpp 的 kDatasetDir / kVectorDir / kGoldenDir（或你改过的环境变量版），不再调用 gen_data.py。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="spmv_custom"
SKIP_BUILD=0
# 与 CMakeLists 一致，可通过环境覆盖： export SOC_VERSION=Ascend310P3
: "${SOC_VERSION:=Ascend910B3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    *)
      echo "ERROR: unknown argument $1" >&2
      echo "用法: $0 [--skip-build]" >&2
      exit 1
      ;;
  esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/3] 设置 CANN 环境 ==="
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
  echo "=== [2/3] 跳过编译（复用已有产物）==="
else
  echo "=== [2/3] 编译 (SOC_VERSION=${SOC_VERSION}) ==="
  mkdir -p build
  ( cd build && cmake -DSOC_VERSION="${SOC_VERSION}" .. && make -j4 ) || die "编译失败"
fi

echo "=== [3/3] 运行 LP（.mps）测试：${OP_NAME} ==="
cd build
mkdir -p output
"./${OP_NAME}" || die "运行 ${OP_NAME} 失败"

echo "=== 完成（结果见 build/output/ 下 CSV 与终端输出）==="
