# cannbot_spmv：SpMV Kernel

AscendC 直调 CSR SpMV 算子样例，计算 `y = A x`。

当前版本：
- 稀疏格式：CSR
- dtype：fp32
- 目标：单机直调可编译、可运行、可校验
- 当前优化：row_ptr 按 tile 批量搬运、col_idx/values 按 chunk 搬运、x 按局部窗口缓存、y 按 tile 批量写回

## 运行

```bash
bash run.sh --shape tiny
bash run.sh --shape small
bash run.sh --shape medium
bash run.sh --shape large
bash run.sh --instance Dual2_5000 --scale 4096
```
