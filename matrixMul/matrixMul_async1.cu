/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */

/* Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication and is exactly the same as
 * Chapter 7 of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * CUBLAS provides high-performance matrix multiplication.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "cuda_safe.h"

// includes, kernels
#include <matrixMul_kernel.cu>

void randomInit(float*, int);
void printDiff(float*, float*, int, int);

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
	unsigned int N= 0;
	float elapsed_time_in_Ms= 0;
	float bandwidth_in_MBs= 0;
	int i, max_iter= 10;
	float *h_A, *h_B, *h_C;
	cudaStream_t stream;

	if( argc > 1 )
		N = atoi( argv[1] );
	else
		N = 1024;

	cudaSetDevice( 0 );
	// set seed for rand()
	srand(2006);

	/* CUDA flags:
	cudaHostAllocDefault, cudaHostAllocPortable, cudaHostAllocMapped,
	cudaHostAllocWriteCombined */
	unsigned int flags= cudaHostAllocDefault;
	// allocate host memory for matrices A and B
	unsigned int size_A = WA * HA;
	unsigned int mem_size_A = sizeof(float) * size_A;
	CUDA_SAFE_CALL( cudaHostAlloc( (void**)&h_A, mem_size_A, flags ) );
	unsigned int size_B = WB * HB;
	unsigned int mem_size_B = sizeof(float) * size_B;
	CUDA_SAFE_CALL( cudaHostAlloc( (void**)&h_B, mem_size_B, flags ) );
	// allocate device memory for result
	unsigned int size_C = WC * HC;
	unsigned int mem_size_C = sizeof(float) * size_C;
	// allocate host memory for the result
	CUDA_SAFE_CALL( cudaHostAlloc( (void**)&h_C, mem_size_C, flags ) );

	cudaEvent_t e1, e2;

	cudaEventCreate( &e1 );
	cudaEventCreate( &e2 );
	cudaStreamCreate( &stream );
	// initialize host memory
	randomInit(h_A, size_A);
	randomInit(h_B, size_B);

	// allocate device memory
	float* d_A;
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_A, mem_size_A));
	float* d_B;
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_B, mem_size_B));
	float* d_C;
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_C, mem_size_C));
	// setup execution parameters
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(WC / threads.x, HC / threads.y);

	CUDA_SAFE_CALL(cudaEventRecord( e1, 0 ));
	for( i= 0; i < max_iter; i++ ){
		// copy host memory to device
		CUDA_SAFE_CALL( cudaMemcpyAsync(d_A, h_A, mem_size_A,
				      cudaMemcpyHostToDevice, stream) );
		CUDA_SAFE_CALL(cudaMemcpyAsync(d_B, h_B, mem_size_B,
				      cudaMemcpyHostToDevice, stream) );
		// execute the kernel
		matrixMul<<< grid, threads, 0, stream >>>(d_C, d_A, d_B, WA,
				WB);
		// check if kernel execution generated and error
		//cutilCheckMsg("Kernel execution failed");
		// copy result from device to host
		CUDA_SAFE_CALL(cudaMemcpyAsync( h_C, d_C, mem_size_C,
				      cudaMemcpyDeviceToHost, stream ) );
		CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );
	}
	CUDA_SAFE_CALL(cudaEventRecord( e2, 0 ));
	CUDA_SAFE_CALL(cudaEventSynchronize( e2 ));
	CUDA_SAFE_CALL(cudaEventElapsedTime( &elapsed_time_in_Ms, e1, e2 ));
	bandwidth_in_MBs= 1e3f * max_iter * (3.0f*N*N*sizeof(float)) / 
	       	(elapsed_time_in_Ms * (float)(1 << 20));
	fprintf( stdout, "async1 size= %d time(ms)= %.3f bandwidth(MB/s)= %.1f\n",
		N, elapsed_time_in_Ms/(max_iter), bandwidth_in_MBs );


	if( argc > 2 ){
		float error_norm;
		float ref_norm;
		float diff;
		// compute reference solution
		float* h_C_ref= (float*) malloc(mem_size_C);
		computeGold(h_C_ref, h_A, h_B, HA, WA, WB);

		/* Check result against reference */
		error_norm = 0;
		ref_norm = 0;
		for (i = 0; i < N; ++i) {
			diff = h_C_ref[i] - h_C[i];
			error_norm += diff * diff;
			ref_norm += h_C_ref[i] * h_C_ref[i];
		}
		error_norm = (float)sqrt((double)error_norm);
		ref_norm = (float)sqrt((double)ref_norm);
		if (fabs(ref_norm) < 1e-7) {
			fprintf (stderr, "!!!! reference norm is 0\n");
			return EXIT_FAILURE;
		}
		printf( "Test %s\n",
			(error_norm / ref_norm < 1e-6f) ? "PASSED" : "FAILED");
		free(h_C_ref);
	}

	// clean up memory
	CUDA_SAFE_CALL( cudaStreamDestroy(stream) );
	CUDA_SAFE_CALL( cudaFreeHost( h_A ) );
	CUDA_SAFE_CALL( cudaFreeHost( h_B ) );
	CUDA_SAFE_CALL( cudaFreeHost( h_C ) );
	CUDA_SAFE_CALL( cudaEventDestroy(e1) );
	CUDA_SAFE_CALL( cudaEventDestroy(e2) );
	CUDA_SAFE_CALL(cudaFree(d_A));
	CUDA_SAFE_CALL(cudaFree(d_B));
	CUDA_SAFE_CALL(cudaFree(d_C));

	cudaThreadExit();
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height)
{
  int i,j,k;
  int error_count=0;
  for (j=0; j<height; j++) {
    for (i=0; i<width; i++) {
      k = j*width+i;
      if (data1[k] != data2[k]) {
         printf("diff(%d,%d) CPU=%4.4f, GPU=%4.4f n", i,j, data1[k], data2[k]);
         error_count++;
      }
    }
  }
  printf(" nTotal Errors = %d n", error_count);
}

