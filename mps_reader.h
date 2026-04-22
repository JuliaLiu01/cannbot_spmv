#ifndef MPS_READER
#define MPS_READER

#include "highs/Highs.h"
#include "highs/util/HighsSparseMatrix.h"
#include "cstring"
#include <iostream>
#include <algorithm>

#define HPR_LP_FLOAT float
using namespace std;

// struct of CSR matrix
// We need these values for constructing CUDA CSR sparse matrix through "cusparseCreateCsr".
struct sparseMatrix {
    int row, col;
    int numElements;
    int *colIndex;
    int *rowPtr;
    HPR_LP_FLOAT *value;
};

// This function plays a role in reading MPS file and formulate the LP problem.
void formulation(const std::string& mps_fp, sparseMatrix *A);

#endif