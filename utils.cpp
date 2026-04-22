#include "utils.h"
#include "mps_reader.h"


const int MAX_BLOCKS = 40;              // Ascend blocks
const int TILE_SIZE = 2048;      


void mergePathSearch(
    int32_t diagonal,
    int32_t nrows,
    int32_t nnz,
    const std::vector<int>& row_ptr,
    int32_t& sx,
    int32_t& sy)
{
    int32_t x_min = std::max(0, diagonal - nnz);
    int32_t x_max = std::min(diagonal, nrows);

    while (x_min < x_max) {
        long long pivot = (x_min + x_max) / 2;
        long long A_val = row_ptr[pivot + 1];
        long long B_val = diagonal - pivot - 1;
        if (A_val <= B_val) {
            x_min = pivot + 1;
        } else {
            x_max = pivot;
        }
    }

    sx = std::min(x_min, nrows);
    sy = diagonal - sx;
}


void CSR_nnz_pad(sparseMatrix &A) {
    int32_t nnz = 0;
    for(int32_t i = 0 ; i < A.row ; ++ i) {
        int32_t num = A.rowPtr[i + 1] - A.rowPtr[i];
        int32_t last = (num + 7) / 8 * 8;

        if(num == 0){
            last = 8;
        }

        nnz += last;
    }

    A.numElements = nnz;

    std::cout << "New nnz number is " << nnz << std::endl;

    HPR_LP_FLOAT *padded_values = (HPR_LP_FLOAT *)malloc(sizeof(HPR_LP_FLOAT)* nnz);
    int *padded_colIdx = (int *)malloc(sizeof(int)* nnz);
    int *padded_rowPtrs =(int *)malloc(sizeof(int)* (A.row + 1));
    padded_rowPtrs[0]=0;

    int32_t nnz_ptr = 0;

    for(int32_t i = 0 ; i < A.row ; ++ i) {
        int32_t num = A.rowPtr[i + 1] - A.rowPtr[i];
        int32_t last = (num + 7) / 8 * 8;

        if(num == 0){
            last = 8;
        }

        int32_t k = 0;
        for(k = 0 ; k < num ; ++ k) {
            padded_values[nnz_ptr] = A.value[A.rowPtr[i] + k];
            padded_colIdx[nnz_ptr] = A.colIndex[A.rowPtr[i] + k];
            nnz_ptr ++;
        }
        for(; k < last ; ++ k) {
            padded_values[nnz_ptr] = 0.0f;
            padded_colIdx[nnz_ptr] = A.col - 1;
            nnz_ptr ++;
        }
        padded_rowPtrs[i + 1] = nnz_ptr;
    }

    free(A.value);
    free(A.colIndex);
    free(A.rowPtr);

    A.rowPtr = new int32_t[A.row + 1];
    A.colIndex = new int32_t[A.numElements];
    A.value = new HPR_LP_FLOAT[A.numElements];

    memcpy(A.value, padded_values, sizeof(HPR_LP_FLOAT) * nnz);
    memcpy(A.colIndex, padded_colIdx, sizeof(int32_t) * nnz);
    memcpy(A.rowPtr, padded_rowPtrs, sizeof(int32_t) * (A.row + 1));
}


void prepareTilingData_simple(
    const sparseMatrix& csrMatrix,
    int32_t& totalTiles,
    std::vector<int32_t>& blockTiles,       // 每个block的tile数量
    std::vector<int32_t>& blockTileOffsets, // 每个block的tile偏移量
    std::vector<int32_t>& tileSx,           // 每个tile的起始行
    std::vector<int32_t>& tileSy,           // 每个tile的起始非零元素
    std::vector<int32_t>& tileEx,           // 每个tile的结束行
    std::vector<int32_t>& tileEy            // 每个tile的结束非零元素
) 
{          
    
    int nrows = csrMatrix.row;
    int nnz = csrMatrix.numElements;
    int total_diagonals = nrows + nnz;
    

    int work_per_block = (total_diagonals + MAX_BLOCKS - 1) / MAX_BLOCKS;
    

    std::vector<int32_t> blockStartRow(MAX_BLOCKS, 0);
    std::vector<int32_t> blockEndRow(MAX_BLOCKS, nrows);
    std::vector<int32_t> blockStartNnz(MAX_BLOCKS, 0);
    std::vector<int32_t> blockEndNnz(MAX_BLOCKS, nnz);
    

    for (int blockId = 0; blockId < MAX_BLOCKS; ++blockId) {

        int startDiagonal = blockId * work_per_block;
        int endDiagonal = std::min(startDiagonal + work_per_block, total_diagonals);
        
        
        int32_t sx_start, sy_start;
        mergePathSearch(startDiagonal, nrows, nnz, 
                           std::vector<int>(csrMatrix.rowPtr, csrMatrix.rowPtr + nrows + 1),
                           sx_start, sy_start);
        
        int32_t sx_end, sy_end;
        mergePathSearch(endDiagonal, nrows, nnz,
                           std::vector<int>(csrMatrix.rowPtr, csrMatrix.rowPtr + nrows + 1),
                           sx_end, sy_end);
        

        sy_start = std::max(sy_start, csrMatrix.rowPtr[sx_start]);
        sy_end = std::min(sy_end, csrMatrix.rowPtr[sx_end + 1]);
        
        blockStartRow[blockId] = sx_start;
        blockEndRow[blockId] = sx_end;
        blockStartNnz[blockId] = sy_start;
        blockEndNnz[blockId] = sy_end;
    }
    
    blockTiles.resize(MAX_BLOCKS, 0);
    blockTileOffsets.resize(MAX_BLOCKS, 0);
    
    // 因为在Kernel内部，由于 UB Memory Constraint
    // 无法把 每一个block的 [sy_start, sy_end) 这块 nnz 进行并行计算
    // 所以这里采用了 二级划分, 把 [sy_start, sy_end) 以 TILE_SIZE 为单位进行二次划分
    for (int blockId = 0; blockId < MAX_BLOCKS; ++blockId) {
        int nnzInBlock = blockEndNnz[blockId] - blockStartNnz[blockId];
        if (nnzInBlock <= 0) {
            blockTiles[blockId] = 0;
            continue;
        }
        blockTiles[blockId] = (nnzInBlock + TILE_SIZE - 1) / TILE_SIZE; // 向上取整
    }
    
    blockTileOffsets[0] = 0;
    for (int blockId = 1; blockId < MAX_BLOCKS; ++blockId) {
        blockTileOffsets[blockId] = blockTileOffsets[blockId - 1] + blockTiles[blockId - 1];
    }
    
    totalTiles = blockTileOffsets[MAX_BLOCKS - 1] + blockTiles[MAX_BLOCKS - 1];
    
    tileSx.resize(totalTiles, 0);
    tileSy.resize(totalTiles, 0);
    tileEx.resize(totalTiles, 0);
    tileEy.resize(totalTiles, 0);
    
    int currentTile = 0;

    for (int blockId = 0; blockId < MAX_BLOCKS; ++blockId) {
        if (blockTiles[blockId] == 0) continue;
        
        int currentRow = blockStartRow[blockId];
        int currentNnz = blockStartNnz[blockId];
        int remainingNnz = blockEndNnz[blockId] - blockStartNnz[blockId];
        
        for (int tileInBlock = 0; tileInBlock < blockTiles[blockId]; ++tileInBlock) {

            // 计算当前tile的非零元素数量
            int nnzInTile = std::min(TILE_SIZE, remainingNnz);
            remainingNnz -= nnzInTile;
            
            // 确定当前tile的结束非零元素位置
            // tile 的工作区间为 [currentNnz, endNnz]
            int endNnz = currentNnz + nnzInTile - 1;
            
            // 确保不会超出block范围
            endNnz = std::min(endNnz, blockEndNnz[blockId] - 1);
            
            // 寻找结束行 - 从当前行开始查找
            int endRow = currentRow;
            while (endRow <= blockEndRow[blockId] && 
                   csrMatrix.rowPtr[endRow + 1] - 1 < endNnz) {
                endRow++;
            }
            
            // 设置tile边界
            tileSx[currentTile] = currentRow;
            tileEx[currentTile] = endRow;
            tileSy[currentTile] = currentNnz;
            tileEy[currentTile] = endNnz;
            
            // 更新为下一个tile准备
            currentRow = endRow;
            currentNnz = tileEy[currentTile] + 1;
            if(currentNnz >= csrMatrix.rowPtr[currentRow + 1]) currentRow += 1;
            ++ currentTile;
        }
    }
}


void gen_random_x(float *x, int32_t len) {

    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    for(int32_t i = 0 ; i < len ; ++ i) {
        float r = 2.0f * static_cast<float>(std::rand()) / RAND_MAX;
        x[i] = r;
    }
}


void cpu_csr_spmv(sparseMatrix *A, float *x, double *y) {
    for(int i = 0 ; i < A->row ; ++ i) {
        double temp = 0.0;
        for(int j = A->rowPtr[i] ; j < A->rowPtr[i + 1] ; ++ j) {
            double val = static_cast<double>(A->value[j]);
            double _x = static_cast<double>(x[A->colIndex[j]]);
            temp += val * _x;
        }
        y[i] = temp;
    }
}


void show_vector(const std::vector<int>& vec, int32_t len) 
{
    for(int i = 0 ; i < len ; i ++) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}


void simple_pc_scaling(sparseMatrix *A, float *colNorm)
{
    //row scaling
    for(int32_t i = 0 ; i < A->row ; i ++) {
        float row_max = std::numeric_limits<float>::min();
        for(int32_t j = A->rowPtr[i] ; j < A->rowPtr[i + 1] ; ++ j) {
            row_max = max(row_max, A->value[j]);
        }

        if(abs(row_max - 0.0f) <= __FLT_EPSILON__) {
            row_max = 1.0;
        }

        for(int32_t j = A->rowPtr[i] ; j < A->rowPtr[i + 1] ; ++ j) {
            A->value[j] = A->value[j] / row_max;
        }
    }

    //column scaling
    for(int32_t i = 0 ; i < A->row ; i ++) {
        for(int32_t j = A->rowPtr[i] ; j < A->rowPtr[i + 1] ; ++ j) {
            A->value[j] = A->value[j] / colNorm[A->colIndex[j]];
        }
    }
}


void CSR_transpose_host(sparseMatrix A, sparseMatrix &AT) {
    AT.row = A.col;
    AT.col = A.row;
    AT.numElements = A.numElements;
    AT.value = new HPR_LP_FLOAT [AT.numElements];
    AT.colIndex = new int [AT.numElements];
    AT.rowPtr = new int [AT.row+2];

    for (int i = 0; i < AT.row+2; i++) {
        AT.rowPtr[i] = 0;
    }

    for (int i = 0; i < A.numElements; i++) {
        AT.rowPtr[A.colIndex[i]+2]++;
    }

    AT.rowPtr[0] = 0;
    for (int i = 2; i < AT.row+2; i++) {
        AT.rowPtr[i] += AT.rowPtr[i-1];
    }

    for (int i = 0; i < A.row; i++) {
        for (int j = A.rowPtr[i]; j < A.rowPtr[i+1]; j++) {
            int col = A.colIndex[j];
            int index = AT.rowPtr[col+1]++;
            AT.value[index] = A.value[j];
            AT.colIndex[index] = i;
        }
    }
}


void row_maximum(const sparseMatrix& A, float *temp) {
    //row scaling
    for(int32_t i = 0 ; i < A.row ; i ++) {
        float row_max = std::numeric_limits<float>::min();
        for(int32_t j = A.rowPtr[i] ; j < A.rowPtr[i + 1] ; ++ j) {
            row_max = max(row_max, A.value[j]);
        }

        if(abs(row_max - 0.0f) <= __FLT_EPSILON__) {
            row_max = 1.0;
        }

        temp[i] = row_max;
    }
}


void ruiz_scaling(sparseMatrix *A, sparseMatrix &AT)
{
    for(int i = 0 ; i < 10 ; ++ i) {

        float *temp1 = (float *)malloc(sizeof(float) * A->row);

        for(int j = 0 ; j < A->row ; ++ j) {
            float row_sum = 0.0f;
            for(int k = A->rowPtr[j] ; k < A->rowPtr[j + 1] ; ++ k) {
                row_sum += abs(A->value[k]);
            }
            if(abs(row_sum - 0.0f) <= __FLT_EPSILON__) {
                row_sum = 1.0;
            }
            temp1[j] = row_sum;

            for(int k = A->rowPtr[j] ; k < A->rowPtr[j + 1] ; ++ k) {
                A->value[k] = A->value[k] / row_sum;
            }
        }

        float *temp2 = (float *)malloc(sizeof(float) * AT.row);

        for(int j = 0 ; j < AT.row ; ++ j) {
            float row_sum = 0.0f;
            for(int k = AT.rowPtr[j] ; k < AT.rowPtr[j + 1] ; ++ k) {
                row_sum += abs(AT.value[k]);
            }
            if(abs(row_sum - 0.0f) <= __FLT_EPSILON__) {
                row_sum = 1.0;
            }
            temp2[j] = row_sum;

            for(int k = AT.rowPtr[j] ; k < AT.rowPtr[j + 1] ; ++ k) {
                AT.value[k] = AT.value[k] / row_sum;
            }
        }


        for(int j = 0 ; j < A->row ; ++ j) {
            for(int k = A->rowPtr[j] ; k < A->rowPtr[j + 1] ; ++ k) {
                A->value[k] = A->value[k] / temp2[A->colIndex[k]];
            }
        }

        for(int j = 0 ; j < AT.row ; ++ j) {
            for(int k = AT.rowPtr[j] ; k < AT.rowPtr[j + 1] ; ++ k) {
                AT.value[k] = AT.value[k] / temp1[AT.colIndex[k]];
            }
        }

        free(temp1);
        free(temp2);
    }
}


// void parse_tile(
//     const sparseMatrix& csrMatrix,
//     const int totalTiles,
//     const std::vector<int32_t> tileSx,
//     const std::vector<int32_t> tileSy,
//     const std::vector<int32_t> tileEx,
//     const std::vector<int32_t> tileEy,

//     std::vector<int32_t>& ordered_nums,     // 每个元素记录 本块 tile 的不同 col 的数量
//     std::vector<int32_t>& ordered_cols,     // 每个元素记录 本块 tile 的不同 col (元素)
//     std::vector<int32_t>& ordered_cols_num,
//     std::vector<float>& ordered_vals,       // 记录每个 col 的对应 nnz value
//     std::vector<int32_t>& ordered_rows,     // 记录每个 col 的对应 nnz row (在本块tile里的相对位置)
//     std::vector<int32_t>& order_tile_offsets1,
//     std::vector<int32_t>& order_tile_offsets2
// ) {

//     order_tile_offsets2.resize(totalTiles, 0);

//     for (int i = 0 ; i < totalTiles ; ++ i) {
//         int sy = tileSy[i];
//         int ey = tileEy[i];

//         vector<int> cols;
//         std::map<int, bool> visited;
//         for(int j = sy ; j <= ey ; ++ j) {
//             int col = csrMatrix.colIndex[j];
//             if (visited[col]) {
//                 continue;
//             }
//             visited[col] = true;
//             cols.push_back(col);
//         }

//         // 判断这些不重复的 col 和 本 tile 段 所有 nnz 数量的关系
//         if (int(cols.size()) < (ey - sy + 1) * 0.4) { 

//             // 把每个 col 对应的 nnz 在本段里的位置放入
//             ordered_nums.push_back(int(cols.size()));
//             for(int j = 0 ; j < cols.size() ; ++ j) {

//                 int num = 0;
//                 for(int k = sy ; k <= ey ; ++ k ) {
//                     if(csrMatrix.colIndex[k] == cols[j]) {
//                         ordered_vals.push_back(csrMatrix.value[k]);
//                         for(int m = tileSx[i] ; m <= tileEx[i] ; ++ m) {
//                             if(k < csrMatrix.rowPtr[m + 1] && k >= csrMatrix.rowPtr[m]) {
//                                 ordered_rows.push_back(k - tileSx[i]);      // 放入的是这个nnz在当前 tile 里的相对行坐标
//                                 break;
//                             }
//                         }
//                         ++ num;
//                     }
//                 }
//                 ordered_cols_num.push_back(num);
//                 ordered_cols.push_back(cols[j]);
//             }
//         }
//         else {
//             ordered_nums.push_back(0);
//         }

//         order_tile_offsets2[i] = 
//     }

//     // 计算 offset
//     order_tile_offsets1.resize(totalTiles, 0);
//     for(int i = 1 ; i < totalTiles ; ++ i) {
//         order_tile_offsets1[i] = order_tile_offsets1[i - 1] + ordered_nums[i - 1];
//     }
// }