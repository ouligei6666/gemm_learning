#include <math.h>
#include <stdio.h>

#define BLOCK_SIZE 2
#define GET_MATRIX_ELEMENT(M, ROW, COL, N) ((M)[(ROW) * (N) + (COL)])

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

void block_matrix_multiply(const double* A, const double* B, double* C, int N) {
    for(int i = 0; i < N*N; i++){
        C[i] = 0.0;
    }
 
    for(int i = 0; i < N; i += BLOCK_SIZE){
        for(int j = 0; j < N; j+= BLOCK_SIZE){
            for(int k = 0; k < N; k += BLOCK_SIZE){
                int ii_end = (i + BLOCK_SIZE < N) ? (i + BLOCK_SIZE) : N;
                int jj_end = (j + BLOCK_SIZE < N) ? (j + BLOCK_SIZE) : N;
                int kk_end = (k + BLOCK_SIZE < N) ? (k + BLOCK_SIZE) : N;
 
                for(int ii = i; ii < ii_end; ii++){
                    for(int jj = j; jj < jj_end; jj++){
                        for(int kk = k; kk < kk_end; kk++){
                            GET_MATRIX_ELEMENT(C, ii, jj, N) +=
                                GET_MATRIX_ELEMENT(A, ii, kk, N) * GET_MATRIX_ELEMENT(B, kk, jj, N);
                        }
                    }
                }
            }
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

    double C_block[N * N];
    double C_ref[N * N];

    block_matrix_multiply(A, B, C_block, N);
    naive_matrix_multiply(A, B, C_ref, N);

    double max_abs_diff = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double diff = fabs(C_block[i] - C_ref[i]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
        }
    }

    printf("max_abs_diff = %.6f\n", max_abs_diff);
    printf("C_block =\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%8.2f ", GET_MATRIX_ELEMENT(C_block, i, j, N));
        }
        printf("\n");
    }

    return 0;
}