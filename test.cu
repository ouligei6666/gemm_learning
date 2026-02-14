#include <stdio.h>

__global__ void hello_from_gpu() {
    printf("Hello from GPU!\n");
}

int main() {
    hello_from_gpu<<<4, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}