#ifndef UTILS
#define UTILS

#include <vector>
#include <iostream>
#include <cstdlib> 
#include <ctime>   
#include <map>
#include "mps_reader.h"



void mergePathSearch(
    int32_t diagonal,
    int32_t nrows,
    int32_t nnz,
    const std::vector<int>& row_ptr,
    int32_t& sx,
    int32_t& sy
);



void prepareTilingData_simple(
    const sparseMatrix& csrMatrix,
    int32_t& totalTiles,
    std::vector<int32_t>& blockTiles,       // 每个block的tile数量
    std::vector<int32_t>& blockTileOffsets, // 每个block的tile偏移量
    std::vector<int32_t>& tileSx,           // 每个tile的起始行
    std::vector<int32_t>& tileSy,           // 每个tile的起始非零元素
    std::vector<int32_t>& tileEx,           // 每个tile的结束行
    std::vector<int32_t>& tileEy            // 每个tile的结束非零元素
);



/*
把每一块 Tile [Sy, Ey) 的 colIndex 聚个类

把 colIdex 相等的 nnz 的在这块 Tile 里的相对坐标整理到 ordered_cols 里面
[同时注意，假如聚类效果很差{最差情况下这块 tile 里的 nnz 全部 col 不相等}，那么不存储聚类结果，仍旧保持原先的scaler col方式]


*/
// void parse_tile(
//     const sparseMatrix& csrMatrix,
//     const int totalTiles,
//     const std::vector<int32_t> tileSx,
//     const std::vector<int32_t> tileSy,
//     const std::vector<int32_t> tileEx,
//     const std::vector<int32_t> tileEy,


//     std::vector<int32_t>& ordered_nums,     // 每个元素记录 本块 tile 的不同 col 的数量
//     std::vector<int32_t>& ordered_cols,     // 每个元素记录 本块 tile 的不同 col (元素)
//     std::vector<int32_t>& ordered_cols_num, // 每个元素记录 本块 tile 的不同 col 所占据的 nnz
//     std::vector<float>& ordered_vals,       // 记录每个 col 的对应 nnz value
//     std::vector<int32_t>& ordered_rows,     // 记录每个 col 的对应 nnz row (在本块tile里的相对位置)
//     std::vector<int32_t>& order_tile_offsets1,
//     std::vector<int32_t>& order_tile_offsets2
// );


void gen_random_x(float *x, int32_t len);

void CSR_nnz_pad(sparseMatrix &A);

void show_vector(const std::vector<int>& vec, int32_t len);

void cpu_csr_spmv(sparseMatrix *A, float *x, double *y);

void simple_pc_scaling(sparseMatrix *A, float *colNorm);

void CSR_transpose_host(sparseMatrix A, sparseMatrix &AT);

void row_maximum(const sparseMatrix& A, float *temp);

void ruiz_scaling(sparseMatrix *A, sparseMatrix &AT);

#endif