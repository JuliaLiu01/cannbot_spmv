#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "acl/acl.h"
#include "mps_reader.h"
#include "utils.h"

int LaunchSpmvKernel(uint32_t blockNum, aclrtStream stream,
                     uintptr_t devRowPtr, uintptr_t devColIdx, uintptr_t devValues,
                     uintptr_t devX, uintptr_t devY, uintptr_t devTiling);

// 与 spmv.asc 中 struct SpmvTilingData 布局一致；放全局供 main 使用
struct SpmvTilingData {
    uint32_t blockNum;
    uint32_t rows;
    uint32_t cols;
    uint32_t nnz;
    uint32_t rowPtrTileRows;
};

constexpr uint32_t kRowPtrTileRows = 128;

namespace {
constexpr char kDatasetDir[] = "/home/Hans49_original/";//lp文件的存储路径
constexpr char kVectorDir[] = "/workspace/data/shared/gen_x/";//x的存储路径
constexpr char kGoldenDir[] = "/workspace/data/shared/cu64f_result/"; //gpu上FP64精度下的计算结果 y=Ax
constexpr char kCsvPath[] = "./output/hans49_benchmark.csv";
constexpr int kRepeatTimes = 10;

int CheckAclRet(aclError ret, const char* expr)
{
    if (ret != ACL_SUCCESS) {
        std::cerr << expr << " failed, ret=" << ret << std::endl;
        return ret;
    }
    return ACL_SUCCESS;
}

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

} // namespace

int main()
{
    auto ret = aclInit(nullptr);
    if (CheckAclRet(ret, "aclInit") != ACL_SUCCESS) {
        return ret;
    }

    int32_t deviceId = 0;
    ret = aclrtSetDevice(deviceId);
    if (CheckAclRet(ret, "aclrtSetDevice") != ACL_SUCCESS) {
        aclFinalize();
        return ret;
    }

    int64_t availableCoreNum = 0;
    ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    if (CheckAclRet(ret, "aclrtGetDeviceInfo") != ACL_SUCCESS) {
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }
    uint32_t blockNum = std::max<uint32_t>(1, static_cast<uint32_t>(availableCoreNum));

    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (CheckAclRet(ret, "aclrtCreateStream") != ACL_SUCCESS) {
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }

    std::ofstream csv(kCsvPath, std::ios::trunc);
    csv << "name,time,err\n";
    csv.close();

    DIR* dir = opendir(kDatasetDir);
    if (dir == nullptr) {
        std::cerr << "failed to open dataset dir: " << kDatasetDir << std::endl;
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ACL_ERROR_FAILURE;
    }

    struct dirent* entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename == "." || filename == ".." || !EndsWith(filename, ".mps")) {
            continue;
        }

        std::string pathname = std::string(kDatasetDir) + filename;
        struct stat statbuf {};
        if (stat(pathname.c_str(), &statbuf) != 0 || !S_ISREG(statbuf.st_mode)) {
            continue;
        }

        std::cout << "testing " << filename << std::endl;

        sparseMatrix matrix {};
        sparseMatrix transpose {};
        formulation(pathname, &matrix);
        CSR_transpose_host(matrix, transpose);

        std::vector<float> colNorm(transpose.row);
        row_maximum(transpose, colNorm.data());
        simple_pc_scaling(&matrix, colNorm.data());

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
            continue;
        }

        if (golden.size() != static_cast<size_t>(matrix.row)) {
            std::cerr << "golden size mismatch: " << golden.size() << " vs " << matrix.row << std::endl;
            FreeMatrix(&matrix);
            FreeMatrix(&transpose);
            continue;
        }
        if (x.size() != static_cast<size_t>(matrix.col)) {
            std::cerr << "x size mismatch: " << x.size() << " vs " << matrix.col << std::endl;
            FreeMatrix(&matrix);
            FreeMatrix(&transpose);
            continue;
        }

        // --- 设备缓冲：分配后一次性 H2D + 传 tiling，均不计时 ---
        const size_t rowPtrBytes = static_cast<size_t>(matrix.row + 1) * sizeof(int32_t);
        const size_t colIdxBytes = static_cast<size_t>(matrix.numElements) * sizeof(int32_t);
        const size_t valuesBytes = static_cast<size_t>(matrix.numElements) * sizeof(float);
        const size_t xBytes = static_cast<size_t>(matrix.col) * sizeof(float);
        const size_t yBytes = static_cast<size_t>(matrix.row) * sizeof(float);
        std::vector<float> output(static_cast<size_t>(matrix.row), 0.0f);

        int32_t* devRowPtr = nullptr;
        int32_t* devColIdx = nullptr;
        float* devValues = nullptr;
        float* devX = nullptr;
        float* devY = nullptr;
        SpmvTilingData* devTiling = nullptr;
        const auto freeDeviceBuffers = [&]() {
            if (devTiling != nullptr) {
                aclrtFree(devTiling);
                devTiling = nullptr;
            }
            if (devY != nullptr) {
                aclrtFree(devY);
                devY = nullptr;
            }
            if (devX != nullptr) {
                aclrtFree(devX);
                devX = nullptr;
            }
            if (devValues != nullptr) {
                aclrtFree(devValues);
                devValues = nullptr;
            }
            if (devColIdx != nullptr) {
                aclrtFree(devColIdx);
                devColIdx = nullptr;
            }
            if (devRowPtr != nullptr) {
                aclrtFree(devRowPtr);
                devRowPtr = nullptr;
            }
        };
        const auto fatalAclAndExit = [&]() {
            freeDeviceBuffers();
            FreeMatrix(&matrix);
            FreeMatrix(&transpose);
            closedir(dir);
            aclrtDestroyStream(stream);
            aclrtResetDevice(deviceId);
            aclFinalize();
        };
        if (CheckAclRet(aclrtMalloc(reinterpret_cast<void**>(&devRowPtr), rowPtrBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc rowPtr") != ACL_SUCCESS) {
            freeDeviceBuffers();
            FreeMatrix(&matrix);
            FreeMatrix(&transpose);
            continue;
        }
        if (CheckAclRet(aclrtMalloc(reinterpret_cast<void**>(&devColIdx), colIdxBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc colIdx") != ACL_SUCCESS) {
            freeDeviceBuffers();
            FreeMatrix(&matrix);
            FreeMatrix(&transpose);
            continue;
        }
        if (CheckAclRet(aclrtMalloc(reinterpret_cast<void**>(&devValues), valuesBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc values") != ACL_SUCCESS) {
            freeDeviceBuffers();
            FreeMatrix(&matrix);
            FreeMatrix(&transpose);
            continue;
        }
        if (CheckAclRet(aclrtMalloc(reinterpret_cast<void**>(&devX), xBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc x") != ACL_SUCCESS) {
            freeDeviceBuffers();
            FreeMatrix(&matrix);
            FreeMatrix(&transpose);
            continue;
        }
        if (CheckAclRet(aclrtMalloc(reinterpret_cast<void**>(&devY), yBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc y") != ACL_SUCCESS) {
            freeDeviceBuffers();
            FreeMatrix(&matrix);
            FreeMatrix(&transpose);
            continue;
        }
        if (CheckAclRet(aclrtMalloc(reinterpret_cast<void**>(&devTiling), sizeof(SpmvTilingData), ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc tiling") != ACL_SUCCESS) {
            freeDeviceBuffers();
            FreeMatrix(&matrix);
            FreeMatrix(&transpose);
            continue;
        }

        // 主机上填写 SpmvTilingData 与 H2D tiling 不计入 kernel 时间
        SpmvTilingData tiling {};
        tiling.blockNum = blockNum;
        tiling.rows = static_cast<uint32_t>(matrix.row);
        tiling.cols = static_cast<uint32_t>(matrix.col);
        tiling.nnz = static_cast<uint32_t>(matrix.numElements);
        tiling.rowPtrTileRows = kRowPtrTileRows;

        ret = aclrtMemcpy(devRowPtr, rowPtrBytes, matrix.rowPtr, rowPtrBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (CheckAclRet(ret, "aclrtMemcpy rowPtr") != ACL_SUCCESS) {
            fatalAclAndExit();
            return ret;
        }
        ret = aclrtMemcpy(devColIdx, colIdxBytes, matrix.colIndex, colIdxBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (CheckAclRet(ret, "aclrtMemcpy colIdx") != ACL_SUCCESS) {
            fatalAclAndExit();
            return ret;
        }
        ret = aclrtMemcpy(devValues, valuesBytes, matrix.value, valuesBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (CheckAclRet(ret, "aclrtMemcpy values") != ACL_SUCCESS) {
            fatalAclAndExit();
            return ret;
        }
        ret = aclrtMemcpy(devX, xBytes, x.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (CheckAclRet(ret, "aclrtMemcpy x") != ACL_SUCCESS) {
            fatalAclAndExit();
            return ret;
        }
        ret = aclrtMemcpy(devY, yBytes, output.data(), yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (CheckAclRet(ret, "aclrtMemcpy y init") != ACL_SUCCESS) {
            fatalAclAndExit();
            return ret;
        }
        ret = aclrtMemcpy(devTiling, sizeof(SpmvTilingData), &tiling, sizeof(SpmvTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
        if (CheckAclRet(ret, "aclrtMemcpy tiling") != ACL_SUCCESS) {
            fatalAclAndExit();
            return ret;
        }

        // 首次跑核 + D2H，用于精度 err（不计入 avg 时间统计）
        ret = LaunchSpmvKernel(blockNum, stream,
            reinterpret_cast<uintptr_t>(devRowPtr), reinterpret_cast<uintptr_t>(devColIdx),
            reinterpret_cast<uintptr_t>(devValues), reinterpret_cast<uintptr_t>(devX),
            reinterpret_cast<uintptr_t>(devY), reinterpret_cast<uintptr_t>(devTiling));
        if (CheckAclRet(ret, "LaunchSpmvKernel (correctness)") != ACL_SUCCESS) {
            fatalAclAndExit();
            return ret;
        }
        ret = aclrtMemcpy(output.data(), yBytes, devY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (CheckAclRet(ret, "aclrtMemcpy y output (correctness)") != ACL_SUCCESS) {
            fatalAclAndExit();
            return ret;
        }

        double err2 = 0.0;
        for (int i = 0; i < matrix.row; ++i) {
            const double diff = static_cast<double>(output[i]) - golden[i];
            err2 += diff * diff;
        }

        // 仅统计 spmv_custom 核（与 aclrtSynchronize）耗时；不含 H2D/D2H/tiling/主机建 tiling
        std::chrono::microseconds totalDuration(0);
        for (int iter = 0; iter < kRepeatTimes; ++iter) {
            const auto start = std::chrono::high_resolution_clock::now();
            ret = LaunchSpmvKernel(blockNum, stream,
                reinterpret_cast<uintptr_t>(devRowPtr), reinterpret_cast<uintptr_t>(devColIdx),
                reinterpret_cast<uintptr_t>(devValues), reinterpret_cast<uintptr_t>(devX),
                reinterpret_cast<uintptr_t>(devY), reinterpret_cast<uintptr_t>(devTiling));
            const auto end = std::chrono::high_resolution_clock::now();
            if (CheckAclRet(ret, "LaunchSpmvKernel (timed)") != ACL_SUCCESS) {
                fatalAclAndExit();
                return ret;
            }
            totalDuration += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        }

        freeDeviceBuffers();

        std::cout << "rows=" << matrix.row
                  << ", cols=" << matrix.col
                  << ", nnz=" << matrix.numElements
                  << ", avg_kernel_only_us=" << totalDuration.count() / kRepeatTimes
                  << ", err=" << err2 << std::endl;

        std::ofstream appendCsv(kCsvPath, std::ios::app);
        appendCsv << filename << ',' << totalDuration.count() / kRepeatTimes << ',' << err2 << '\n';

        FreeMatrix(&matrix);
        FreeMatrix(&transpose);
        std::cout << std::endl;
    }

    closedir(dir);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ACL_SUCCESS;
}
