


# 公式：
$C[i][j]=\sum_{k=0}^{N-1}A[i][k]*B[k][j]$

C的第i行第j列的元素=A的第i行与B的第j列做乘累加


# GEMM naive版本

三重循环即可，参考[gemm_naive](gemm_naive.cpp)

# 分块思想

实现在[gemm_block](gemm_block.cpp)

一般数组在存储的时候是按照行主序存储的，对于矩阵A访问时连续，缓存命中率更高，而B矩阵按列访问，在内存中跨步stride，命中低

- 分块之后矩阵更小可以装入cache，集市不能完全装入也比大矩阵的装入命中率高

- 一个A的块可以与多个B的块配对，反复使用



# 矩阵转置：

实现在[gemm_transpose](gemm_transpose.cpp)

将矩阵B的列转化为行，可以解决局部性问题

为什么naive版本比转置版本还要快？

矩阵太小N=4 时，乘法本身几乎没有缓存瓶颈，转置的收益体现不出来，反而多了额外访存和循环开销。


# cpu中的SIMD指令集


实现在[gemm_simd](gemm_simd.cpp)

列出支持SIMD的指令集

`lscpu | grep 'Flags:' | grep -oE '\<sse[0-9]*\>|\<avx[0-9]*\>|\<fma[0-9]*/>'`


参考文献：
\[1\] https://blog.csdn.net/Sunine_686/article/details/149275141 