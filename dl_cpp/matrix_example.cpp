#include <Eigen/Dense>
#include <iostream>

int main() {
    // Define two 3x3 matrices initialized with values
    Eigen::Matrix3d matrixA;
    Eigen::Matrix3d matrixB;

    matrixA << 1, 2, 3,
               4, 5, 6,
               7, 8, 9;

    matrixB << 9, 8, 7,
               6, 5, 4,
               3, 2, 1;

    // Matrix addition
    Eigen::Matrix3d sum = matrixA + matrixB;
    std::cout << "Sum of matrixA and matrixB:\n" << sum << std::endl;

    // Matrix multiplication
    Eigen::Matrix3d prod = matrixA * matrixB;
    std::cout << "Product of matrixA and matrixB:\n" << prod << std::endl;

    // Solve linear system Ax = b
    Eigen::Vector3d b(3, 3, 3);
    Eigen::Vector3d x = matrixA.colPivHouseholderQr().solve(b);
    std::cout << "Solution of Ax = b, x:\n" << x << std::endl;

    // Compute eigenvalues
    Eigen::EigenSolver<Eigen::Matrix3d> eigensolver(matrixA);
    if (eigensolver.info() != Eigen::Success) {
        std::cerr << "Eigenvalue computation failed!" << std::endl;
        return 1;
    }
    std::cout << "Eigenvalues of matrixA:\n" << eigensolver.eigenvalues() << std::endl;

    return 0;
}
