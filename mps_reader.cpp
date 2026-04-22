#include "mps_reader.h"

void sparse_csc_to_csr(vector<int>& csr_row_start, vector<int>& csr_col_indices, 
                        vector<HPR_LP_FLOAT>& csr_values, vector<int>& row_counts,
                        const HighsSparseMatrix& matrix, int m, int n){ 
    
    std::cout << "Converting from CSC format to CSR format." << std::endl;
    for (int col = 0; col < n; ++col) {
        // traverse each column 
        for (int k = matrix.start_[col]; k < matrix.start_[col + 1]; ++k) {
            // get the row index of the nnz value in this column by {matrix.index_[k]}
            // update the number of nnz values this row has
            row_counts[matrix.index_[k]]++;
        }
    }

    // Compute row_start
    std::partial_sum(row_counts.begin(), row_counts.end(), csr_row_start.begin() + 1);

    // Fill col_indices and values
    std::vector<int> row_pos(m, 0);     // a relative position of each nnz value in its row
    for (int col = 0; col < n; ++col) {
        for (int k = matrix.start_[col]; k < matrix.start_[col + 1]; ++k) {
            int row = matrix.index_[k];

            // the true position that this value in the CSR layout
            int pos = csr_row_start[row] + row_pos[row];    

            csr_col_indices[pos] = col;
            csr_values[pos] = static_cast<HPR_LP_FLOAT>(matrix.value_[k]);
            row_pos[row]++;
        }
    }
}

void safe_double_to_float(std::vector<HPR_LP_FLOAT>& values, const std::vector<double>& dvals) {
    for(auto element : dvals) {
        values.push_back(static_cast<HPR_LP_FLOAT>(element));
    }
}

void formulation(const std::string& mps_fp, sparseMatrix *A) {
    Highs highs;
    // Set HiGHS to run quietly (optional, reduces console output)
    highs.setOptionValue("output_flag", false);
    // Read the MPS file
    std::printf("Start reading file....\n");
    HighsStatus read_status = highs.readModel(mps_fp);
    std::printf("Reading complete....\n");

    // Get model information (e.g., number of variables and constraints)
    const HighsModel& model = highs.getModel();
    int m = model.lp_.num_row_;
    int n = model.lp_.num_col_;
    int nnz = static_cast<int>(model.lp_.a_matrix_.value_.size());

    // The CSR sparse matrix descriptor. 
    std::vector<int> row_start;
    std::vector<int> col_indices;
    std::vector<HPR_LP_FLOAT> values;
    HighsSparseMatrix matrix = model.lp_.a_matrix_;
    std::vector<int> row_counts(m, 0);                              // record the number of nnz values each row
    if (model.lp_.a_matrix_.format_ == MatrixFormat::kRowwise) {
        // Default in CSR format
        // values = model.lp_.a_matrix_.value_;
        safe_double_to_float(values, model.lp_.a_matrix_.value_);
        row_start = model.lp_.a_matrix_.start_;
        col_indices = model.lp_.a_matrix_.index_;

        for (int row = 0 ; row < m ; ++row) {
            for (int k = row_start[row] ; k < row_start[row + 1] ; ++k){
                row_counts[row] ++;
            }
        }
    }
    // ------- Notice that the read in matrix maybe in CSC format ------- //
    else if (model.lp_.a_matrix_.format_ == MatrixFormat::kColwise) {
        // Initialize CSR arrays. Directly allocate memory.
        row_start.resize(m + 1, 0);                         
        col_indices.resize(nnz);                            
        values.resize(nnz);

        sparse_csc_to_csr(row_start, col_indices, values, row_counts, model.lp_.a_matrix_, m, n);
    }

    // std::vector<HPR_LP_FLOAT> row_lower = model.lp_.row_lower_;
    // std::vector<HPR_LP_FLOAT> row_upper = model.lp_.row_upper_;
    std::vector<HPR_LP_FLOAT> row_lower, row_upper;
    safe_double_to_float(row_lower, model.lp_.row_lower_);
    safe_double_to_float(row_upper, model.lp_.row_upper_);


    // Filtering the all-zero rows and unconstrained (-inf <= Ax <= inf) rows
    std::vector<int> kept_row_inds;
    int empty_rows = 0;
    int nnz_reduced = 0;
    for (int index = 0 ; index < m ; ++index) {
        if(row_counts[index] == 0 || (row_lower[index] == -kHighsInf && row_upper[index] == kHighsInf)){
            ++empty_rows;
            if(row_lower[index] == -kHighsInf && row_upper[index] == kHighsInf) {
                nnz_reduced += row_counts[index];
            }
        }
        else kept_row_inds.push_back(index);
    }

    std::cout << "Deleted " << m - kept_row_inds.size() << " unuseful rows\n";

    nnz -= nnz_reduced;

    bool flag = (kept_row_inds.size() == m);        // Do not have redundant things [no need to reconstruct A]
    vector<HPR_LP_FLOAT> new_row_upper(kept_row_inds.size());
    vector<HPR_LP_FLOAT> new_row_lower(kept_row_inds.size());
    vector<int> new_col_indices;
    vector<int> new_row_start(kept_row_inds.size() + 1);
    vector<HPR_LP_FLOAT> new_values;
    new_col_indices.reserve(nnz);
    new_values.reserve(nnz);
    int new_row_idx = 0;
    int nz_count = 0;

    if(flag) {
        std::copy(row_upper.begin(), row_upper.end(), new_row_upper.begin());
        std::copy(row_lower.begin(), row_lower.end(), new_row_lower.begin());
        std::copy(row_start.begin(), row_start.end(), new_row_start.begin());
        std::copy(col_indices.begin(), col_indices.end(), new_col_indices.begin());
        std::copy(values.begin(), values.end(), new_values.begin());
    }

    // Helper function to copy/negate row [Reason is that we may have to delete some rows from original A!]
    auto copyRow = [&](int rowId) {
        new_row_upper[new_row_idx] = row_upper[rowId];
        new_row_lower[new_row_idx] = row_lower[rowId];
        for (int k = row_start[rowId]; k < row_start[rowId + 1]; ++k) {
            new_col_indices.push_back(col_indices[k]);
            new_values.push_back(values[k]);
            nz_count++;
        }
        new_row_idx++;
        new_row_start[new_row_idx] = nz_count;
    };


    // Get the index of the different types of constraints
    int eq_constraints_num = 0;
    for (const int non_empty_index : kept_row_inds) {    // traverse the non-empty rows is enough
        if (row_lower[non_empty_index] == row_upper[non_empty_index]) {
            //  equality constraint
            eq_constraints_num += 1;
        } 
        if(!flag) copyRow(non_empty_index);
    }

    std::cout << "problem information: nRow = " << kept_row_inds.size() << ", nCol = " << n  << ", nnz A = " << nnz << std::endl;
    std::cout << "number of equalities = " << eq_constraints_num << std::endl;
    std::cout << "number of inequalities = " << kept_row_inds.size() - eq_constraints_num << std::endl;



    // 1. sparse A
    A->row = kept_row_inds.size();
    A->col = n;
    A->numElements = nnz;
    A->colIndex = new int[nnz];
    A->value = new HPR_LP_FLOAT[nnz];
    A->rowPtr = new int[A->row + 1];

    memcpy(A->value, new_values.data(), sizeof(HPR_LP_FLOAT) * nnz);
    memcpy(A->colIndex, new_col_indices.data(), sizeof(int) * nnz);
    memcpy(A->rowPtr, new_row_start.data(), sizeof(int) * (A->row + 1));

}

