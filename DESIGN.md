# SpMV 算法设计文档

## 1. 目标

当前算子实现 AscendC 直调版 CSR SpMV，计算目标为：

```text
y = A x
```

其中：
- `A`：稀疏矩阵，采用 CSR 存储
- `x`：fp32 稠密向量
- `y`：fp32 输出向量

设计目标分为三层：
1. 单机直调可编译、可运行、可校验
2. 支持大规模、低密度稀疏矩阵场景
3. 在不改变 CSR 接口的前提下，逐步消除明显的低效访存路径

## 2. 输入输出与数据格式

### 2.1 输入

当前 Host 侧输入文件为：
- `input/row_ptr.bin`：长度 `rows + 1`，类型 `int32`
- `input/col_idx.bin`：长度 `nnz`，类型 `int32`
- `input/values.bin`：长度 `nnz`，类型 `float32`
- `input/x.bin`：长度 `cols`，类型 `float32`

其中 CSR 语义如下：
- `row_ptr[i]` 到 `row_ptr[i + 1]` 给出第 `i` 行非零元区间
- `col_idx[k]` 表示第 `k` 个非零元所在列号
- `values[k]` 表示第 `k` 个非零元数值

### 2.2 输出

- `output/output.bin`：长度 `rows`，类型 `float32`
- `output/golden.bin`：NumPy 生成的参考结果，用于精度校验

## 3. 总体实现结构

代码主文件为 `spmv_kernel/spmv.asc`，整体分为两部分：

1. **Device 侧 Kernel**
   - `spmv_custom(...)`
   - `KernelSpmv::Init()`
   - `KernelSpmv::Process()`

2. **Host 侧直调逻辑**
   - ACL 初始化、设备设置、Stream 创建
   - 输入数据读取、Device 内存申请与拷贝
   - tiling 参数构造
   - kernel 启动与结果回拷

## 4. Host 侧设计

### 4.1 启动参数

当前 Host 侧通过 `main()` 接收：
- `rows`
- `cols`
- `nnz`

并基于设备 `VECTOR_CORE_NUM` 构造 `SpmvTilingData`：

```cpp
struct SpmvTilingData {
    uint32_t blockNum;
    uint32_t rows;
    uint32_t cols;
    uint32_t nnz;
    uint32_t rowPtrTileRows;
};
```

含义如下：
- `blockNum`：实际启动的核数
- `rowPtrTileRows`：kernel 内部一次处理的行块大小

### 4.2 分核策略

当前采用**按 blockIdx 映射的连续行分块**策略：
- Host 侧仍只下发 `blockNum`
- Device 侧用 `rows * blockIdx / blockNum` 和 `rows * (blockIdx + 1) / blockNum` 计算当前 block 的行区间
- 不改变 CSR 格式，不做 host 端重排
- 优点是实现简单、边界清晰、便于正确性验证
- 缺点是不同 row 的 nnz 差异大时，负载仍可能不均衡

该策略仍是保守选择，但比固定 `rowsPerCore`/尾块处理更紧凑，避免单独维护尾块逻辑。

## 5. Kernel 数据流设计

### 5.1 常量配置

当前 kernel 关键 tile 参数：

```cpp
constexpr uint32_t OUTPUT_TILE_ROWS = 128;
constexpr uint32_t ROW_PTR_TILE_ROWS = 128;
constexpr uint32_t ROW_PTR_TILE_ELEMS = ROW_PTR_TILE_ROWS + 1;
constexpr uint32_t NNZ_CHUNK_ELEMS = 256;
constexpr uint32_t X_WINDOW_MAX_ELEMS = 512;
```

含义：
- `ROW_PTR_TILE_ROWS=128`：一次处理 128 行的 `row_ptr` 信息
- `NNZ_CHUNK_ELEMS=256`：对单行的非零元分块处理，每次搬运 256 个 nnz
- `X_WINDOW_MAX_ELEMS=512`：当 chunk 内列跨度不超过 512 时，将对应 `x` 子段连续搬入 UB
- `OUTPUT_TILE_ROWS=128`：输出 `y` 先在 UB 中累积，再按 128 行批量回写

### 5.2 初始化阶段

`KernelSpmv::Init()` 完成以下工作：
- 解析 tiling 参数
- 计算当前 core 的 `rowStart_ / rowEnd_`
- 绑定 `rowPtr / colIdx / values / x / y` 的 GlobalTensor
- 初始化 `TQue`

当前使用的队列/缓冲：
- `rowPtrQueue_`
- `colIdxQueue_`
- `valuesQueue_`
- `outQueueY_`
- `xWindowBuf_`

用途：
- `rowPtrQueue_`：承接当前 row tile 的 `row_ptr` 批量搬运
- `colIdxQueue_ / valuesQueue_`：承接从 GM 搬到 UB 的连续 nnz chunk
- `outQueueY_`：承接当前 row tile 的输出缓存
- `xWindowBuf_`：承接 chunk 内局部列窗口对应的连续 `x` 子段

### 5.3 主处理流程

`Process()` 的执行逻辑如下：

#### 步骤 1：确定本核负责的行区间

若当前 core 对应区间为空，直接返回。

#### 步骤 2：按 row tile 处理

外层循环以 `rowPtrTileRows` 为步长推进：
- 当前 tile 覆盖 `[rowBase, rowBase + rowsThisTile)`
- 先通过 `DataCopyPad` 将该 tile 需要的 `row_ptr` 信息批量搬入 UB

这样做的目的，是去掉 `row_ptr` 的逐元素 GM 访问。

#### 步骤 3：为输出申请 UB 缓冲

为当前 row tile 分配 `yLocal`：
- 初始全部置零
- 每完成一行累加后，将结果写入 `yLocal[localRow]`

#### 步骤 4：逐行计算

对 tile 内每一行：
- 通过 `rowPtrTile[localRow]` 和 `rowPtrTile[localRow + 1]` 取得该行的 nnz 区间
- 用 `sum` 保存该行累加结果
- 若该行 nnz 超过 `NNZ_CHUNK_ELEMS`，则按 chunk 逐段处理

#### 步骤 5：按 nnz chunk 搬运 `col_idx / values`

`CopyInChunk(chunkStart, chunkElems)` 会：
- 从 GM 连续拷贝 `col_idx[chunkStart:chunkStart+chunkElems]`
- 从 GM 连续拷贝 `values[chunkStart:chunkStart+chunkElems]`
- 将两者分别入队

这里已经从“逐元素 GM 读取”优化为“连续段批量搬运到 UB”。

#### 步骤 6：在 UB 上遍历 chunk，并按需缓存局部 `x` 窗口

`ComputeChunk(chunkElems)` 中：
- 从队列中取出 `colLocal` 和 `valuesLocal`
- 先扫描 chunk 内列号，求 `minCol/maxCol`
- 若列跨度不超过 `X_WINDOW_MAX_ELEMS`，则将 `x[minCol:maxCol]` 连续搬入 `xWindowBuf_`
- 后续优先从局部 `x` 窗口读取 `xLocal[col - minCol]`
- 若列跨度过大，则退回 `xGm_.GetValue(col)`

计算语义等价于：

```text
sum += values[k] * x[col_idx[k]]
```

#### 步骤 7：批量写回输出

当前 row tile 全部计算完成后：
- 将 `yLocal` 入队
- 通过 `CopyOutBatch()` 一次性 `DataCopyPad` 写回 `yGm_[rowBase]`

这一步避免了“每行一次单独回写”的高同步开销。

## 6. 两轮演进与优化思路

### 6.1 第一阶段：正确性优先版本

最初版本重点是先建立完整链路：
- CSR 输入
- Host 直调
- NumPy golden 校验
- Kernel 正确输出

该阶段主要解决了两个问题：
1. 先把 SpMV 样例完整跑通
2. 修复 `GlobalTensor::SetValue()` 直接写输出导致结果异常的问题，改为 `LocalTensor + DataCopyPad` 写回

### 6.2 第二阶段：保守优化

在不改 CSR 接口、不重写整体结构的前提下，进行了保守优化：

#### 优化 1：row_ptr 小 tile 化

将 `row_ptr` 的使用从分散读取收敛到按 tile 处理：
- 每次处理 128 行
- 先读取这一小段 row 边界
- 降低了主循环中的行边界访问成本

#### 优化 2：col_idx / values 分块搬运

将连续的 nnz 段从 GM 批量拷入 UB：
- 替换掉大量逐元素 `GetValue()` 访问
- 更符合 AscendC 推荐的数据搬运方式

#### 优化 3：输出批量回写

将输出从“逐行写回”改为“tile 累积后批量写回”：
- 减少 `DataCopyPad` 次数
- 减少同步点
- 提升写回阶段效率

### 6.3 第三阶段：row_ptr 批量搬运与 x 局部窗口缓存

本轮继续保持 CSR 接口不变，增加两项保守优化：

#### 优化 1：row_ptr 批量搬运到 UB

将 tile 所需的 `row_ptr` 边界通过 `DataCopyPad` 一次搬入 `rowPtrQueue_`：
- 去掉 `rowPtrGm_.GetValue()` 的逐元素慢路径
- 让行边界读取也统一到批量搬运模型

#### 优化 2：chunk 级 x 局部窗口缓存

对每个 nnz chunk：
- 先估算该 chunk 的 `minCol/maxCol`
- 若列跨度足够小，则把对应 `x` 子段连续搬入 `xWindowBuf_`
- 后续乘加从 UB 读取局部 `x`，减少随机 GM 访问次数
- 若列跨度过大，则退回旧路径，避免为离散列分布引入额外开销

## 7. 当前性能特征与瓶颈

当前版本已经消除了最明显的几类低效点，但仍存在核心瓶颈。

### 7.1 已改善的部分

- `row_ptr` 已改成按 tile 批量搬运到 UB
- `col_idx / values` 改成按 chunk 连续搬运到 UB
- `y` 输出改成批量写回
- 当 chunk 内列跨度较小时，`x` 可按局部窗口搬入 UB
- 整体数据流从“逐元素 GM 读写”演进为“连续数据批量搬运 + 局部计算”

### 7.2 当前主要瓶颈

当前最主要的瓶颈仍然是：

```cpp
float xValue = xGm_.GetValue(col);
```

原因：
- `x` 的访问由 `col_idx` 决定，本质上是随机访问
- 对极大规模稀疏矩阵，列索引离散性较强
- 即使 `col_idx / values` 已经连续化，`x` 仍难以天然形成大块连续搬运

这意味着当前版本仍然是：
- **row_ptr / col_idx / values / y** 路径已经做了第一轮优化
- **x** 路径仍是最主要的访存热点

## 8. 后续可继续优化的方向

如果继续做第二阶段以上优化，可考虑以下方向：

### 8.1 基于列局部性的 x 缓存

对 chunk 内涉及到的列号做局部分析：
- 若列号集中，可先把对应 `x` 子段搬入 UB
- 再在 UB 中完成乘加

难点在于：
- chunk 内列号未必连续
- 去重、排序、索引映射都会引入额外开销

### 8.2 长短行分治

针对 nnz 分布高度不均的矩阵：
- 长行单独处理
- 短行继续按当前 row tile 方案处理

这样可以降低某些长行对单 core 执行时间的拖累。

### 8.3 按 nnz 而非按 row 分核

当前按行均分实现简单，但在行长度差异大时利用率不稳定。
后续可考虑：
- 基于 `row_ptr` 统计每个 core 的 nnz 预算
- 做更均衡的任务划分

代价是 host 端 tiling 逻辑会更复杂。

### 8.4 更激进的数据重排

若允许改变输入布局，可进一步考虑：
- 分块 CSR
- SELL-C/σ
- 按列局部性重排

但这已经超出当前“保持 CSR 接口不变”的范围。

## 9. 测试与验证策略

### 9.1 合成 shape

当前 `scripts/gen_data.py` 提供以下标准 shape：
- `tiny`
- `small`
- `medium`
- `large`

这些 shape 用于：
- 正确性回归
- 越界检查
- 不同规模下的基本冒烟测试

### 9.2 基于真实矩阵信息的缩放测试

当前支持从 `待测试的A矩阵基本信息.csv` 读取真实矩阵元信息：
- 行数
- 列数
- 非零元数量
- 稠密度

再通过 `--scale` 对规模做缩放，保留稀疏度特征，用于构造更接近真实分布的测试数据。

示例：

```bash
bash run.sh --instance Dual2_5000 --scale 4096
```

该方式的价值在于：
- 不必直接生成原始超大矩阵
- 仍可保留真实场景中的行列规模比例与稀疏度特征

### 9.3 golden 校验

参考结果由 NumPy 生成：

```python
golden[row] = np.dot(values[start:end], x[col_idx[start:end]])
```

再通过 `verify_result.py` 校验 `output/output.bin` 与 `output/golden.bin` 一致性。

## 10. 当前结论

当前 SpMV 算子已经具备以下能力：
- CSR fp32 输入输出链路完整
- Host 直调与 ACL 运行流程完整
- 可通过标准 shape 与真实矩阵缩放用例做正确性验证
- 已完成一轮面向数据搬运路径的保守优化

当前版本适合作为：
- AscendC 直调 CSR SpMV 的可运行样例
- 后续更深层性能优化的基线版本

其核心特点可以概括为：
- **接口保持简单**：不改变 CSR 格式
- **实现保持保守**：先优化连续数据搬运路径
- **验证链路完整**：支持 golden 校验和规模扩展
- **瓶颈清晰可见**：后续重点集中在 `x` 的随机访存优化与更均衡的分核策略
