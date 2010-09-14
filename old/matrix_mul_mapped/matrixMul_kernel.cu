
#ifndef _DEVICE_KERNEL_H_
#define _DEVICE_KERNEL_H_

#include <stdio.h>
#include "device.h"

extern "C"
__global__ void matrixMul( int *C, int *A, int *B, int N )
{
	int value = 0;
	int i;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	for( i = 0; i < N; i++ )
		value += A[row * N + i] * B[i * N + col];
	C[row * N + col] = value;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
