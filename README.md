# cannbot_spmv：SpMV Kernel

AscendC 直调 CSR SpMV 算子样例，计算 **y = A x**。主机侧读 LP/MPS 问题中的约束矩阵 A（CSR），在 NPU 上执行核函数，并与 golden 对比误差；可输出 kernel-only 耗时（不含 H2D/D2H 与 tiling 主机准备时间）。

## 功能概览

- **稀疏格式**：CSR（fp32）
- **设备核**：`spmv.asc` 中的 `spmv_custom`（经 `LaunchSpmvKernel` 启动）
- **构建**：`find_package(ASC)` + 单可执行目标 `spmv_custom`（`spmv.asc` + `main.cpp` 等）
- **运行模式**（见 `main.cpp`）：
  - **`Test()`**：遍历 `kDatasetDir` 下全部 `*.mps` 做基准
  - **`Analysis()`**：只跑代码里写死的若干算例（见 `Analysis` 内 `kCases` 列表，会随开发调整）
  - 当前 **`main` 默认调用 `Analysis()`**；若需全量目录测试，将 `main` 中改为调用 `Test(stream, blockNum)`（并视编译器提示处理未使用函数）

## 依赖

- 已安装 **CANN / ascend-toolkit**，并配置 **`ASCEND_HOME_PATH`**
- 链接 **HiGHS**（`CMakeLists.txt` 中 `/usr/local/lib64/libhighs.so` 等，按本机安装路径修改）
- 编译时 **`SOC_VERSION`** 与目标 NPU/仿真一致（如 `Ascend910B3`），可通过 `cmake -DSOC_VERSION=...` 或缓存指定

## 数据路径（在 `main.cpp` 顶部常量中配置）

| 常量 | 含义 |
|------|------|
| `kDatasetDir` | 存放 `*.mps` |
| `kVectorDir` | 与算例同名的 `x` 向量，`<case>.txt`（须以 `/` 结尾再拼文件名） |
| `kGoldenDir` | 参考解 `y`，`<case>.txt` |
| `kCsvPath` | 汇总 CSV，默认 `./output/hans49_benchmark.csv`（相对**运行时的当前工作目录**） |

## 编译

```bash
export ASCEND_HOME_PATH=/path/to/ascend-toolkit/latest
source "${ASCEND_HOME_PATH}/set_env.sh"   # 或你环境中的 set_env

mkdir -p build && cd build
cmake -DSOC_VERSION=Ascend910B3 ..
make -j4
```

## 运行

**推荐（一键：环境 + 编译 + 在 `build` 下执行）：**

```bash
bash run.sh
# 仅跳过编译：bash run.sh --skip-build
```

`run.sh` 会在 `build` 目录下执行 `./spmv_custom`，请保证 `main.cpp` 中数据路径与机器一致；结果 CSV 与终端输出中的时间/误差以**在 `build` 下运行**时的相对路径为准。

**手动：**

```bash
source .../set_env.sh
cd build
mkdir -p output
./spmv_custom
```

## 输出说明

- 终端会打印 `deviceId`、`blockNum`、每个算例的 `rows/cols/nnz`、`avg_kernel_only_us`、累计误差等
- CSV 表头为 `name,time,err`，与 `kCsvPath` 对应

## 实现要点（与 DESIGN 一致时）

- row_ptr 按 tile 批量搬运，col_idx / values 按 chunk，x 按局部窗口缓存，y 按 tile 写回

## 仓库

上游示例仓库：<https://github.com/JuliaLiu01/cannbot_spmv>
