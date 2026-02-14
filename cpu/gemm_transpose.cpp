#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

static double seconds_since(const struct timespec* start, const struct timespec* end) {
    double sec = (double)(end->tv_sec - start->tv_sec);
    double nsec = (double)(end->tv_nsec - start->tv_nsec) / 1e9;
    return sec + nsec;
}

int main() {
    const int sizes[] = {256, 512, 1024};
    const int sizes_count = (int)(sizeof(sizes) / sizeof(sizes[0]));

    for (int s = 0; s < sizes_count; ++s) {
        const int N = sizes[s];
        const int iters = (N <= 256) ? 20 : (N <= 512 ? 5 : 2);
        const size_t total = (size_t)N * (size_t)N;

        double* A = (double*)malloc(sizeof(double) * total);
        double* B = (double*)malloc(sizeof(double) * total);
        double* B_T = (double*)malloc(sizeof(double) * total);
        double* C_transposed = (double*)malloc(sizeof(double) * total);
        double* C_ref = (double*)malloc(sizeof(double) * total);

        if (!A || !B || !B_T || !C_transposed || !C_ref) {
            printf("allocation failed for N=%d\n", N);
            free(A);
            free(B);
            free(B_T);
            free(C_transposed);
            free(C_ref);
            return 1;
        }

        for (size_t i = 0; i < total; ++i) {
            A[i] = (double)(i % 13) / 13.0;
            B[i] = (double)(i % 7) / 7.0;
        }

        transpose_matrix(B, B_T, N);

        struct timespec t0;
        struct timespec t1;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int it = 0; it < iters; ++it) {
            naive_matrix_multiply(A, B, C_ref, N);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double naive_sec = seconds_since(&t0, &t1);

        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int it = 0; it < iters; ++it) {
            naive_transposed_matrix_multiply(A, B_T, C_transposed, N);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double transpose_sec = seconds_since(&t0, &t1);

        double max_abs_diff = 0.0;
        for (size_t i = 0; i < total; ++i) {
            double diff = fabs(C_transposed[i] - C_ref[i]);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
            }
        }

        printf("N=%d iters=%d\n", N, iters);
        printf("max_abs_diff = %.6f\n", max_abs_diff);
        printf("naive_time_sec = %.6f (avg %.6f)\n", naive_sec, naive_sec / iters);
        printf("transpose_time_sec = %.6f (avg %.6f)\n", transpose_sec, transpose_sec / iters);

        free(A);
        free(B);
        free(B_T);
        free(C_transposed);
        free(C_ref);
    }

    return 0;
}