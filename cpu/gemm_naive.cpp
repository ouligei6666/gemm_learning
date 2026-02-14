//naive matrix multiplication in C++

#include <stdio.h>

void naive_matrix_multiply(const double* A, const double* B, double* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    const int N = 2;
    const double A[N * N] = {
        1.0, 2.0,
        3.0, 4.0
    };
    const double B[N * N] = {
        5.0, 6.0,
        7.0, 8.0
    };
    double C[N * N] = {0.0};

    naive_matrix_multiply(A, B, C, N);

    printf("C =\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%6.2f ", C[i * N + j]);
        }
        printf("\n");
    }

    return 0;
}