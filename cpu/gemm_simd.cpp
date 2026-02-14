
#pragma GCC target("avx2,fma") //无论命令行参数如何都能编译。如果不加，就用这个g++ -O3 -mavx2 -mfma gemm_simd.cpp -o gemm_simd
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 64
#define GET_MATRIX_ELEMENT(M, ROW, COL, N) ((M)[(ROW) * (N) + (COL)])

typedef union Vec4d {
    __m256d v;
    double d[4];
} Vec4d;

//分块的SIMD实现方式 —— 将尾部数据的处理直接写回到矩阵C中
void SIMD_matrix_multiply_imm(const double* A, const double* B, double* C, int N){
    for (int i = 0; i < N * N; i++){
        C[i] = 0.0;
    }
                        
    for(int i = 0; i < N; i += BLOCK_SIZE){
        for(int j = 0; j < N; j += BLOCK_SIZE){
            for(int k = 0; k < N; k += BLOCK_SIZE){
 
                int end_ii = i + BLOCK_SIZE < N ? i + BLOCK_SIZE : N;
                int end_jj = j + BLOCK_SIZE < N ? j + BLOCK_SIZE : N;
                int end_kk = k + BLOCK_SIZE < N ? k + BLOCK_SIZE : N;
                for(int ii = i; ii < end_ii; ++ii){
                    __m256d c_row_vec[(BLOCK_SIZE + 3) / 4];
                        //初始化累加向量，__m256d寄存器256位，可以存储4个double类型的数据，（上面这个是要上取整）
                    for(int vec_idx = 0; vec_idx < (BLOCK_SIZE + 3) / 4; vec_idx ++){
                        c_row_vec[vec_idx] = _mm256_setzero_pd();
                    }
                    for(int kk = k; kk < end_kk; kk++){
                        __m256d a_val_vec = _mm256_set1_pd(GET_MATRIX_ELEMENT(A,ii,kk,N));
                        int jj_SIMD_limit = j + (end_jj - j) / 4 * 4;
                        for(int jj = j; jj < jj_SIMD_limit; jj += 4){
                            // _mm256_loadu_pd() 用于非对齐加载，更安全但可能稍慢
                            // _mm256_load_pd() 用于对齐加载（如果确保内存对齐则更快）
                            __m256d b_vec = _mm256_loadu_pd(&GET_MATRIX_ELEMENT(B,kk,jj,N));
 
                            int vec_idx = (jj - j) / 4;
                            c_row_vec[vec_idx] = _mm256_fmadd_pd(a_val_vec, b_vec, c_row_vec[vec_idx]);
 
                        }
 
                        for(int jj_tail = jj_SIMD_limit; jj_tail < end_jj; jj_tail ++){
                            GET_MATRIX_ELEMENT(C, ii, jj_tail, N) += GET_MATRIX_ELEMENT(A, ii, kk, N) * GET_MATRIX_ELEMENT(B, kk, jj_tail, N);
 
                        }
                    }
 
                    for(int vec_idx = 0; vec_idx < (end_jj - j) / 4; vec_idx ++){
                        int jj_inner = j + vec_idx * 4;
                        //累加上之前的元素
                        __m256d c_old = _mm256_loadu_pd(&GET_MATRIX_ELEMENT(C,ii,jj_inner,N));
                        __m256d c_new = _mm256_add_pd(c_old, c_row_vec[vec_idx]);
                        _mm256_storeu_pd(&GET_MATRIX_ELEMENT(C,ii,jj_inner,N), c_new);
                    }
                    
                    
                }
            }
        }
    }
}


//分块的SIMD实现方式 —— 先写回到向量中，在最后一起写回到矩阵中
void SIMD_matrix_multiply_vec(const double* A, const double* B, double* C, int N){
    for (int i = 0; i < N*N; i++){
        C[i] = 0.0;
    }
                        
    for(int i = 0; i < N; i += BLOCK_SIZE){
        for(int j = 0; j < N; j += BLOCK_SIZE){
            for(int k = 0; k < N; k += BLOCK_SIZE){
 
                int end_ii = i + BLOCK_SIZE < N ? i + BLOCK_SIZE : N;
                int end_jj = j + BLOCK_SIZE < N ? j + BLOCK_SIZE : N;
                int end_kk = k + BLOCK_SIZE < N ? k + BLOCK_SIZE : N;
 
                int current_block_size = end_jj - j;
                int vec_number_per_block = (current_block_size + 3) / 4;
                for(int ii = i; ii < end_ii; ++ii){
                    Vec4d c_row_vec_union[vec_number_per_block];
                        //初始化累加向量
                    for(int vec_idx = 0; vec_idx < vec_number_per_block; vec_idx ++){
                        c_row_vec_union[vec_idx].v = _mm256_setzero_pd();
                    }
                    for(int kk = k; kk < end_kk; kk++){
                        __m256d a_val_vec = _mm256_set1_pd(GET_MATRIX_ELEMENT(A,ii,kk,N));
                        int jj_SIMD_limit = j + (end_jj - j) / 4 * 4;
                        for(int jj = j; jj < jj_SIMD_limit; jj += 4){
                            // _mm256_loadu_pd() 用于非对齐加载，更安全但可能稍慢
                            // _mm256_load_pd() 用于对齐加载（如果确保内存对齐则更快）
                            __m256d b_vec = _mm256_loadu_pd(&GET_MATRIX_ELEMENT(B,kk,jj,N));
 
                            int vec_idx = (jj - j) / 4;
                            c_row_vec_union[vec_idx].v = _mm256_fmadd_pd(a_val_vec, b_vec, c_row_vec_union[vec_idx].v);
 
                        }
 
                        for(int jj_tail = jj_SIMD_limit; jj_tail < end_jj; jj_tail ++){
                            int vec_idx_tail = (jj_tail - j) / 4;
                            int idx_in_vec = (jj_tail - j) % 4;
                            
                            c_row_vec_union[vec_idx_tail].d[idx_in_vec] += GET_MATRIX_ELEMENT(A,ii,kk,N) * GET_MATRIX_ELEMENT(B,kk,jj_tail,N);
 
                        }
                    }
                    for(int vec_idx = 0; vec_idx < vec_number_per_block; vec_idx ++){
                        int jj_store = j + vec_idx * 4;
                        if(jj_store < j + (end_jj - j)/4 * 4){
                            __m256d c_existing = _mm256_loadu_pd(&GET_MATRIX_ELEMENT(C,ii,jj_store,N));
                            c_existing = _mm256_add_pd(c_existing, c_row_vec_union[vec_idx].v);
                            _mm256_storeu_pd(&GET_MATRIX_ELEMENT(C,ii,jj_store,N), c_existing);
                        }
                        else{
                            for(int idx_in_vec = 0; idx_in_vec < end_jj - jj_store; idx_in_vec ++){
                                GET_MATRIX_ELEMENT(C, ii, jj_store + idx_in_vec, N) += c_row_vec_union[vec_idx].d[idx_in_vec];
                            }
                        }
                    }
                    
                }
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
    const int N = 256;
    const int iters = 5;
    const size_t total = (size_t)N * (size_t)N;

    double* A = (double*)malloc(sizeof(double) * total);
    double* B = (double*)malloc(sizeof(double) * total);
    double* C_imm = (double*)malloc(sizeof(double) * total);
    double* C_vec = (double*)malloc(sizeof(double) * total);
    double* C_ref = (double*)malloc(sizeof(double) * total);

    if (!A || !B || !C_imm || !C_vec || !C_ref) {
        printf("allocation failed\n");
        free(A);
        free(B);
        free(C_imm);
        free(C_vec);
        free(C_ref);
        return 1;
    }

    for (size_t i = 0; i < total; ++i) {
        A[i] = (double)(i % 13) / 13.0;
        B[i] = (double)(i % 7) / 7.0;
    }

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
        SIMD_matrix_multiply_imm(A, B, C_imm, N);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double imm_sec = seconds_since(&t0, &t1);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < iters; ++it) {
        SIMD_matrix_multiply_vec(A, B, C_vec, N);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double vec_sec = seconds_since(&t0, &t1);

    double max_diff_imm = 0.0;
    double max_diff_vec = 0.0;
    for (size_t i = 0; i < total; ++i) {
        double diff_imm = fabs(C_imm[i] - C_ref[i]);
        double diff_vec = fabs(C_vec[i] - C_ref[i]);
        if (diff_imm > max_diff_imm) {
            max_diff_imm = diff_imm;
        }
        if (diff_vec > max_diff_vec) {
            max_diff_vec = diff_vec;
        }
    }

    printf("N=%d iters=%d\n", N, iters);
    printf("max_diff_imm = %.6f\n", max_diff_imm);
    printf("max_diff_vec = %.6f\n", max_diff_vec);
    printf("naive_time_sec = %.6f (avg %.6f)\n", naive_sec, naive_sec / iters);
    printf("imm_time_sec = %.6f (avg %.6f)\n", imm_sec, imm_sec / iters);
    printf("vec_time_sec = %.6f (avg %.6f)\n", vec_sec, vec_sec / iters);

    free(A);
    free(B);
    free(C_imm);
    free(C_vec);
    free(C_ref);
    return 0;
}