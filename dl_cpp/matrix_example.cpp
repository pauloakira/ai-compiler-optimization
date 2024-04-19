extern "C" {
// LAPACK routine for solving systems of linear equations
void dgesv_(int* n, int* nrhs, double* a, int* lda, int* ipiv,
            double* b, int* ldb, int* info);
}

#include <iostream>
#include <vector>

int main() {
    int n = 3; // Dimension of the matrix
    int nrhs = 1; // Number of right-hand sides
    int lda = 3; // Leading dimension of the matrix A
    int ldb = 3; // Leading dimension of the matrix B
    int info; // Output info

    std::vector<int> ipiv(n, 0); // Pivot indices
    std::vector<double> a = {3, 1, 2, 1, 2, 3, 3, 1, 4}; // Matrix A
    std::vector<double> b = {1, 2, 3}; // Matrix B

    // Solve the linear equations A * X = B
    dgesv_(&n, &nrhs, a.data(), &lda, ipiv.data(), b.data(), &ldb, &info);

    if (info == 0) {
        for (int i = 0; i < n; i++) {
            std::cout << "Solution [" << i << "] = " << b[i] << std::endl;
        }
    } else {
        std::cout << "An error occurred: " << info << std::endl;
    }

    return 0;
}
