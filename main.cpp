#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "acl/acl.h"
#include "mps_reader.h"
#include "utils.h"

int LaunchSpmvKernel(uint32_t blockNum, aclrtStream stream,
                     uintptr_t devRowPtr, uintptr_t devColIdx, uintptr_t devValues,
                     uintptr_t devX, uintptr_t devY, uintptr_t devTiling,
                     uintptr_t devProfile);

/*
 * 主流程与函数关系（入口：main）：
 *
 *   main
 *     ├─ ACL 初始化、选 device、建 stream、写 CSV 表头
 *     ├─ 无参 / 非 "analysis" → Test(stream, blockNum)     // 目录下全部 .mps
 *     └─ 首参 "analysis"     → Analysis(stream, blockNum)  // 仅 cont1.mps、cont11.mps
 *
 *   Test / Analysis 均通过 RunSingleMpsCase 处理单个用例（读 mps、x/golden、H2D、核计时、写一行 CSV）
 *
 * 匿名命名空间内：EndsWith、Load*Vector、FreeMatrix、RunSingleMpsCase、Test、Analysis
 */

// 与 spmv.asc 中 struct SpmvTilingData 布局一致；放全局供 host 侧使用
struct SpmvTilingData {
    uint32_t blockNum;
    uint32_t rows;
    uint32_t cols;
    uint32_t nnz;
    uint32_t rowPtrTileRows;
    uint64_t blockRowRanges;
};

struct SpmvBlockProfile {
    uint32_t rowStart;
    uint32_t rowEnd;
    uint32_t rowCount;
    uint32_t nnzCount;
    uint32_t chunkCount;
    uint32_t useWindowCount;
    uint32_t shortRowCount;
};

constexpr uint32_t kRowPtrTileRows = 128;
constexpr uint32_t kShortRowThreshold = 4;

namespace {
struct MatrixProfile {
    double avgNnzPerRow;
    uint32_t maxNnzPerRow;
    double shortRowRatio;
};

struct ProfileSummary {
    uint64_t totalChunks;
    uint64_t useWindowChunks;
    uint64_t totalProfiledNnz;
    uint64_t totalShortRows;
    uint32_t blockRowsMin;
    uint32_t blockRowsMax;
    double blockRowsAvg;
    uint32_t blockNnzMin;
    uint32_t blockNnzMax;
    double blockNnzAvg;
};

struct BlockRowRange {
    uint32_t rowStart;
    uint32_t rowEnd;
};

MatrixProfile BuildMatrixProfile(const sparseMatrix& matrix)
{
    MatrixProfile profile {0.0, 0, 0.0};
    if (matrix.row <= 0) {
        return profile;
    }

    uint64_t totalNnz = 0;
    uint64_t shortRows = 0;
    for (int row = 0; row < matrix.row; ++row) {
        const uint32_t nnz = static_cast<uint32_t>(matrix.rowPtr[row + 1] - matrix.rowPtr[row]);
        totalNnz += nnz;
        profile.maxNnzPerRow = std::max(profile.maxNnzPerRow, nnz);
        if (nnz <= kShortRowThreshold) {
            ++shortRows;
        }
    }

    profile.avgNnzPerRow = static_cast<double>(totalNnz) / matrix.row;
    profile.shortRowRatio = static_cast<double>(shortRows) / matrix.row;
    return profile;
}

ProfileSummary BuildProfileSummary(const std::vector<SpmvBlockProfile>& profiles)
{
    ProfileSummary summary {};
    if (profiles.empty()) {
        return summary;
    }

    summary.blockRowsMin = std::numeric_limits<uint32_t>::max();
    summary.blockNnzMin = std::numeric_limits<uint32_t>::max();
    uint64_t totalRows = 0;
    uint64_t totalNnz = 0;
    for (const SpmvBlockProfile& profile : profiles) {
        summary.totalChunks += profile.chunkCount;
        summary.useWindowChunks += profile.useWindowCount;
        summary.totalShortRows += profile.shortRowCount;
        summary.blockRowsMin = std::min(summary.blockRowsMin, profile.rowCount);
        summary.blockRowsMax = std::max(summary.blockRowsMax, profile.rowCount);
        summary.blockNnzMin = std::min(summary.blockNnzMin, profile.nnzCount);
        summary.blockNnzMax = std::max(summary.blockNnzMax, profile.nnzCount);
        totalRows += profile.rowCount;
        totalNnz += profile.nnzCount;
    }

    summary.totalProfiledNnz = totalNnz;
    summary.blockRowsAvg = static_cast<double>(totalRows) / profiles.size();
    summary.blockNnzAvg = static_cast<double>(totalNnz) / profiles.size();
    if (summary.blockRowsMin == std::numeric_limits<uint32_t>::max()) {
        summary.blockRowsMin = 0;
    }
    if (summary.blockNnzMin == std::numeric_limits<uint32_t>::max()) {
        summary.blockNnzMin = 0;
    }
    return summary;
}

std::vector<BlockRowRange> BuildBalancedRowRanges(const sparseMatrix& matrix, uint32_t blockNum)
{
    std::vector<BlockRowRange> ranges(blockNum, {0, 0});
    if (blockNum == 0 || matrix.row <= 0) {
        return ranges;
    }

    const uint64_t totalNnz = static_cast<uint64_t>(matrix.numElements);
    const uint64_t targetNnzPerBlock = std::max<uint64_t>(1, (totalNnz + blockNum - 1) / blockNum);
    uint32_t currentBlock = 0;
    uint32_t rowStart = 0;
    uint64_t accumulatedNnz = 0;

    for (uint32_t row = 0; row < static_cast<uint32_t>(matrix.row); ++row) {
        const uint32_t rowNnz = static_cast<uint32_t>(matrix.rowPtr[row + 1] - matrix.rowPtr[row]);
        const uint64_t nextAccumulatedNnz = accumulatedNnz + rowNnz;
        const uint32_t remainingRows = static_cast<uint32_t>(matrix.row) - row;
        const uint32_t remainingBlocks = blockNum - currentBlock;
        const bool shouldSplit = currentBlock + 1 < blockNum
            && row >= rowStart
            && accumulatedNnz > 0
            && nextAccumulatedNnz > targetNnzPerBlock
            && remainingRows >= remainingBlocks;
        if (shouldSplit) {
            ranges[currentBlock] = {rowStart, row};
            ++currentBlock;
            rowStart = row;
            accumulatedNnz = 0;
        }
        accumulatedNnz += rowNnz;
    }

    if (currentBlock < blockNum) {
        ranges[currentBlock] = {rowStart, static_cast<uint32_t>(matrix.row)};
        ++currentBlock;
    }
    while (currentBlock < blockNum) {
        ranges[currentBlock] = {static_cast<uint32_t>(matrix.row), static_cast<uint32_t>(matrix.row)};
        ++currentBlock;
    }
    return ranges;
}
constexpr char kDatasetDir[] = "/home/Hans49_original/";//lp文件的存储路径
constexpr char kVectorDir[] = "/workspace/data/shared/gen_x/";//x的存储路径
constexpr char kGoldenDir[] = "/workspace/data/shared/cu64f_result/"; //gpu上FP64精度下的计算结果 y=Ax
constexpr char kCsvPath[] = "./output/hans49_benchmark.csv";
constexpr int kRepeatTimes = 10;

bool EndsWith(const std::string& value, const std::string& suffix)
{
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<float> LoadFloatVector(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open " + path);
    }

    std::vector<float> data;
    float value = 0.0f;
    while (file >> value) {
        data.push_back(value);
    }
    return data;
}

std::vector<double> LoadDoubleVector(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open " + path);
    }

    std::vector<double> data;
    double value = 0.0;
    while (file >> value) {
        data.push_back(value);
    }
    return data;
}

void FreeMatrix(sparseMatrix* matrix)
{
    if (matrix == nullptr) {
        return;
    }
    delete[] matrix->colIndex;
    delete[] matrix->rowPtr;
    delete[] matrix->value;
    matrix->colIndex = nullptr;
    matrix->rowPtr = nullptr;
    matrix->value = nullptr;
}

// 单个用例：kDatasetDir/filename，读 x/golden、跑 SpMV 核、kernel-only 计时、追加 kCsvPath
void RunSingleMpsCase(const std::string& filename, aclrtStream stream, uint32_t blockNum)
{
    if (!EndsWith(filename, ".mps")) {
        return;
    }
    const std::string pathname = std::string(kDatasetDir) + filename;
    struct stat statbuf {};
    if (stat(pathname.c_str(), &statbuf) != 0 || !S_ISREG(statbuf.st_mode)) {
        std::cerr << "skip (not found or not a file): " << pathname << std::endl;
        return;
    }

    std::cout << "testing " << filename << std::endl;

    sparseMatrix matrix {};
    sparseMatrix transpose {};
    formulation(pathname, &matrix);
    CSR_transpose_host(matrix, transpose);

    std::vector<float> colNorm(transpose.row);
    row_maximum(transpose, colNorm.data());
    simple_pc_scaling(&matrix, colNorm.data());
    const MatrixProfile matrixProfile = BuildMatrixProfile(matrix);

    const std::string caseName = filename.substr(0, filename.size() - 4);
    std::vector<float> x;
    std::vector<double> golden;
    try {
        x = LoadFloatVector(std::string(kVectorDir) + caseName + ".txt");
        golden = LoadDoubleVector(std::string(kGoldenDir) + caseName + ".txt");
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        FreeMatrix(&matrix);
        FreeMatrix(&transpose);
        return;
    }

    if (golden.size() != static_cast<size_t>(matrix.row)) {
        std::cerr << "golden size mismatch: " << golden.size() << " vs " << matrix.row << std::endl;
        FreeMatrix(&matrix);
        FreeMatrix(&transpose);
        return;
    }
    if (x.size() != static_cast<size_t>(matrix.col)) {
        std::cerr << "x size mismatch: " << x.size() << " vs " << matrix.col << std::endl;
        FreeMatrix(&matrix);
        FreeMatrix(&transpose);
        return;
    }

    const size_t rowPtrBytes = static_cast<size_t>(matrix.row + 1) * sizeof(int32_t);
    const size_t colIdxBytes = static_cast<size_t>(matrix.numElements) * sizeof(int32_t);
    const size_t valuesBytes = static_cast<size_t>(matrix.numElements) * sizeof(float);
    const size_t xBytes = static_cast<size_t>(matrix.col) * sizeof(float);
    const size_t yBytes = static_cast<size_t>(matrix.row) * sizeof(float);
    const std::vector<BlockRowRange> blockRowRanges = BuildBalancedRowRanges(matrix, blockNum);
    const size_t blockRowRangesBytes = static_cast<size_t>(blockNum) * sizeof(BlockRowRange);
    const size_t profileBytes = static_cast<size_t>(blockNum) * sizeof(SpmvBlockProfile);
    const size_t profileScalarCount = static_cast<size_t>(blockNum) * 7;
    std::vector<float> output(static_cast<size_t>(matrix.row), 0.0f);
    std::vector<uint32_t> hostProfileScalars(profileScalarCount, 0);
    std::vector<SpmvBlockProfile> hostProfiles(blockNum);

    int32_t* devRowPtr = nullptr;
    int32_t* devColIdx = nullptr;
    float* devValues = nullptr;
    float* devX = nullptr;
    float* devY = nullptr;
    BlockRowRange* devBlockRowRanges = nullptr;
    SpmvTilingData* devTiling = nullptr;
    SpmvBlockProfile* devProfile = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&devRowPtr), rowPtrBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&devColIdx), colIdxBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&devValues), valuesBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&devX), xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&devY), yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&devBlockRowRanges), blockRowRangesBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&devTiling), sizeof(SpmvTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&devProfile), profileBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    const auto freeDeviceBuffers = [&]() {
        aclrtFree(devProfile);
        aclrtFree(devTiling);
        aclrtFree(devBlockRowRanges);
        aclrtFree(devY);
        aclrtFree(devX);
        aclrtFree(devValues);
        aclrtFree(devColIdx);
        aclrtFree(devRowPtr);
    };

    SpmvTilingData tiling {};
    tiling.blockNum = blockNum;
    tiling.rows = static_cast<uint32_t>(matrix.row);
    tiling.cols = static_cast<uint32_t>(matrix.col);
    tiling.nnz = static_cast<uint32_t>(matrix.numElements);
    tiling.rowPtrTileRows = kRowPtrTileRows;
    tiling.blockRowRanges = reinterpret_cast<uint64_t>(devBlockRowRanges);

    aclrtMemcpy(devRowPtr, rowPtrBytes, matrix.rowPtr, rowPtrBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(devColIdx, colIdxBytes, matrix.colIndex, colIdxBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(devValues, valuesBytes, matrix.value, valuesBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(devX, xBytes, x.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(devY, yBytes, output.data(), yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(devBlockRowRanges, blockRowRangesBytes, blockRowRanges.data(), blockRowRangesBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(devTiling, sizeof(SpmvTilingData), &tiling, sizeof(SpmvTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemset(devProfile, profileBytes, 0, profileBytes);

    LaunchSpmvKernel(blockNum, stream,
        reinterpret_cast<uintptr_t>(devRowPtr), reinterpret_cast<uintptr_t>(devColIdx),
        reinterpret_cast<uintptr_t>(devValues), reinterpret_cast<uintptr_t>(devX),
        reinterpret_cast<uintptr_t>(devY), reinterpret_cast<uintptr_t>(devTiling),
        reinterpret_cast<uintptr_t>(devProfile));
    aclrtMemcpy(output.data(), yBytes, devY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hostProfileScalars.data(), profileBytes, devProfile, profileBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    for (uint32_t i = 0; i < blockNum; ++i) {
        const size_t base = static_cast<size_t>(i) * 7;
        hostProfiles[i].rowStart = hostProfileScalars[base + 0];
        hostProfiles[i].rowEnd = hostProfileScalars[base + 1];
        hostProfiles[i].rowCount = hostProfileScalars[base + 2];
        hostProfiles[i].nnzCount = hostProfileScalars[base + 3];
        hostProfiles[i].chunkCount = hostProfileScalars[base + 4];
        hostProfiles[i].useWindowCount = hostProfileScalars[base + 5];
        hostProfiles[i].shortRowCount = hostProfileScalars[base + 6];
    }

    double err2 = 0.0;
    for (int i = 0; i < matrix.row; ++i) {
        const double diff = static_cast<double>(output[i]) - golden[i];
        err2 += diff * diff;
    }

    std::chrono::microseconds totalDuration(0);
    for (int iter = 0; iter < kRepeatTimes; ++iter) {
        std::fill(output.begin(), output.end(), 0.0f);
        aclrtMemcpy(devY, yBytes, output.data(), yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemset(devProfile, profileBytes, 0, profileBytes);
        const auto start = std::chrono::high_resolution_clock::now();
        LaunchSpmvKernel(blockNum, stream,
            reinterpret_cast<uintptr_t>(devRowPtr), reinterpret_cast<uintptr_t>(devColIdx),
            reinterpret_cast<uintptr_t>(devValues), reinterpret_cast<uintptr_t>(devX),
            reinterpret_cast<uintptr_t>(devY), reinterpret_cast<uintptr_t>(devTiling),
            reinterpret_cast<uintptr_t>(devProfile));
        const auto end = std::chrono::high_resolution_clock::now();
        totalDuration += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    }

    const ProfileSummary profileSummary = BuildProfileSummary(hostProfiles);
    const double useWindowRatio = profileSummary.totalChunks == 0
        ? 0.0
        : static_cast<double>(profileSummary.useWindowChunks) / profileSummary.totalChunks;

    freeDeviceBuffers();

    std::cout << "rows=" << matrix.row
              << ", cols=" << matrix.col
              << ", nnz=" << matrix.numElements
              << ", blocks=" << blockNum
              << ", avg_kernel_only_us=" << totalDuration.count() / kRepeatTimes
              << ", err=" << err2
              << ", avg_nnz_per_row=" << matrixProfile.avgNnzPerRow
              << ", max_nnz_per_row=" << matrixProfile.maxNnzPerRow
              << ", short_row_ratio=" << matrixProfile.shortRowRatio << std::endl;
    std::cout << "profile total_chunks=" << profileSummary.totalChunks
              << ", use_window_ratio=" << useWindowRatio
              << ", profiled_nnz=" << profileSummary.totalProfiledNnz
              << ", total_short_rows=" << profileSummary.totalShortRows
              << ", block_rows[min,max,avg]=[" << profileSummary.blockRowsMin
              << ',' << profileSummary.blockRowsMax
              << ',' << profileSummary.blockRowsAvg << "]"
              << ", block_nnz[min,max,avg]=[" << profileSummary.blockNnzMin
              << ',' << profileSummary.blockNnzMax
              << ',' << profileSummary.blockNnzAvg << "]" << std::endl;

    std::ofstream appendCsv(kCsvPath, std::ios::app);
    appendCsv << filename << ','
              << totalDuration.count() / kRepeatTimes << ','
              << err2 << ','
              << matrixProfile.avgNnzPerRow << ','
              << matrixProfile.maxNnzPerRow << ','
              << matrixProfile.shortRowRatio << ','
              << profileSummary.totalChunks << ','
              << useWindowRatio << ','
              << profileSummary.totalProfiledNnz << ','
              << profileSummary.totalShortRows << ','
              << profileSummary.blockRowsMin << ','
              << profileSummary.blockRowsMax << ','
              << profileSummary.blockRowsAvg << ','
              << profileSummary.blockNnzMin << ','
              << profileSummary.blockNnzMax << ','
              << profileSummary.blockNnzAvg << '\n';

    FreeMatrix(&matrix);
    FreeMatrix(&transpose);
    std::cout << std::endl;
}

// 全量测试：扫描 kDatasetDir 下所有 .mps 并逐个 RunSingleMpsCase。返回 0 成功，1 表示目录打不开
int Test(aclrtStream stream, uint32_t blockNum)
{
    DIR* dir = opendir(kDatasetDir);
    if (dir == nullptr) {
        std::cerr << "failed to open dataset dir: " << kDatasetDir << std::endl;
        return 1;
    }
    struct dirent* entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        const std::string filename = entry->d_name;
        if (filename == "." || filename == ".." || !EndsWith(filename, ".mps")) {
            continue;
        }
        RunSingleMpsCase(filename, stream, blockNum);
    }
    closedir(dir);
    return 0;
}

// 分析模式：只跑固定列表 cont1.mps、cont11.mps（路径仍为 kDatasetDir + 文件名）
void Analysis(aclrtStream stream, uint32_t blockNum)
{
    const std::vector<std::string> kCases = {
        // "degme.mps",
        // "thk_63.mps",
        // "stp3d.mps",
        
        //---vect慢
        "cont1.mps",// 939
        "neos.mps",//1704
        "graph40-40_lp.mps", //1876

        //---vect快
        "stat96v2.mps",  //1615.67
        "ns168892.mps",  //973.01
    };
    std::cout << "=== Analysis: " << kCases.size() << " selected cases ===\n";
    for (const std::string& name : kCases) {
        RunSingleMpsCase(name, stream, blockNum);
    }
}

} // namespace

// 入口：./spmv_custom 为 Test()；./spmv_custom analysis 为 Analysis()
int main(int argc, char* argv[])
{
    aclInit(nullptr);
    const int32_t deviceId = 7;
    aclrtSetDevice(deviceId);
    int64_t availableCoreNum = 0;
    aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    const uint32_t blockNum = std::max<uint32_t>(1, static_cast<uint32_t>(availableCoreNum));
    std::cout << "deviceId=" << deviceId << ", blockNum=" << blockNum << std::endl;

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    std::ofstream csv(kCsvPath, std::ios::trunc);
    csv << "name,time,err,avg_nnz_per_row,max_nnz_per_row,short_row_ratio,total_chunks,use_window_ratio,profiled_nnz,total_short_rows,block_rows_min,block_rows_max,block_rows_avg,block_nnz_min,block_nnz_max,block_nnz_avg\n";
    csv.close();

    Analysis(stream, blockNum);
    // Test(stream, blockNum);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ACL_SUCCESS;
}
