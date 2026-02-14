#include <math.h>
#include <stdio.h>

#define GET_MATRIX_ELEMENT(M, ROW, COL, N) ((M)[(ROW) * (N) + (COL)])

//矩阵转置的实现
void transpose_matrix(const double* original_matrix, double* transposed_matrix, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            // B_T[j][i] = B[i][j]
            // 对应到一维数组索引：
            // transposed_matrix[j * N + i] = original_matrix[i * N + j];
            GET_MATRIX_ELEMENT(transposed_matrix, j, i, N) = GET_MATRIX_ELEMENT(original_matrix, i ,j, N); 
        }
    }
}

//在naive版本上对矩阵进行转置
void naive_transposed_matrix_multiply(const double* A, const double* B, double* C, int N){
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            GET_MATRIX_ELEMENT(C, i, j, N) = 0.0;
            for (int k = 0; k < N; ++k) {
                GET_MATRIX_ELEMENT(C, i, j, N) += GET_MATRIX_ELEMENT(A, i, k, N) * GET_MATRIX_ELEMENT(B, j, k, N);
            }
        }
    }
}

static void naive_matrix_multiply(const double* A, const double* B, double* C, int N) {
    for (int i = 0; i < N * N; ++i) {
        C[i] = 0.0;
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += GET_MATRIX_ELEMENT(A, i, k, N) * GET_MATRIX_ELEMENT(B, k, j, N);
            }
            GET_MATRIX_ELEMENT(C, i, j, N) = sum;
        }
    }
}

int main() {
    const int N = 4;
    const double A[N * N] = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0
    };
    const double B[N * N] = {
        1.0, 0.0, 2.0, 0.0,
        0.0, 1.0, 0.0, 2.0,
        3.0, 0.0, 4.0, 0.0,
        0.0, 3.0, 0.0, 4.0
    };

    double B_T[N * N];
    double C_transposed[N * N];
    double C_ref[N * N];

    transpose_matrix(B, B_T, N);
    naive_transposed_matrix_multiply(A, B_T, C_transposed, N);
    naive_matrix_multiply(A, B, C_ref, N);

    double max_abs_diff = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double diff = fabs(C_transposed[i] - C_ref[i]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
        }
    }

    printf("max_abs_diff = %.6f\n", max_abs_diff);
    printf("C_transposed =\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%8.2f ", GET_MATRIX_ELEMENT(C_transposed, i, j, N));
        }
        printf("\n");
    }

    return 0;
}