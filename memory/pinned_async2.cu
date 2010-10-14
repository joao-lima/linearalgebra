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
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "add_kernel.cu"
#include "cuda_safe.h"

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
	unsigned int mem_size= (1 << 25);
	float elapsed_time_in_Ms= 0;
	float bandwidth_in_MBs= 0;
	unsigned int i, j, max_iter= 10;
	float *h_data, *d_data;
#define NSTREAM		4
	cudaStream_t stream[NSTREAM];

	if( argc > 1 )
		mem_size =  (1 << atoi(argv[1]));

	unsigned int nelem= mem_size/sizeof(float);
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	for( int d= 0; d < deviceCount; d++ ) {
	cudaSetDevice( d );
	/* CUDA flags:
	cudaHostAllocDefault, cudaHostAllocPortable, cudaHostAllocMapped,
	cudaHostAllocWriteCombined */
	unsigned int flags= cudaHostAllocDefault;
	// allocate host memory for matrices A and B
	CUDA_SAFE_CALL( cudaHostAlloc( (void**)&h_data, mem_size, flags ) );
	for( i= 0; i < nelem; i++) h_data[i]= 1e0f;
	// allocate device memory
	CUDA_SAFE_CALL( cudaMalloc((void**)&d_data, mem_size) );
	cudaEvent_t e1, e2;
	cudaEventCreate( &e1 );
	cudaEventCreate( &e2 );
	for( j= 0; j < NSTREAM; j++ )
		cudaStreamCreate( &stream[j] );
	// setup execution parameters
	//dim3 threads( BLOCK_SIZE, 1 );
	//dim3 grid( 128, 1);
	// number of elements per thread
	unsigned int n_per_stream = nelem / NSTREAM;
	//unsigned int nblock = n_per_stream/(BLOCK_SIZE*grid.x);

	CUDA_SAFE_CALL( cudaEventRecord( e1, 0 ) );
	for( i= 0; i < max_iter; i++ ){
		for( j= 0; j < NSTREAM; j++ ){
		CUDA_SAFE_CALL( cudaMemcpyAsync( d_data+j*n_per_stream,
			h_data+j*n_per_stream,
			n_per_stream*sizeof(float),
			cudaMemcpyHostToDevice, stream[j]) );
		//add_one<<< grid, threads, 0, stream[j] >>>(
		//		d_data+j*n_per_stream, nblock );
		CUDA_SAFE_CALL( cudaMemcpyAsync( h_data+j*n_per_stream,
					d_data+j*n_per_stream,
					n_per_stream*sizeof(float),
				      cudaMemcpyDeviceToHost, stream[j]) );
		}
		cudaThreadSynchronize();
	}
	CUDA_SAFE_CALL( cudaEventRecord( e2, 0 ) );
	CUDA_SAFE_CALL( cudaEventSynchronize( e2 ) );
	CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsed_time_in_Ms, e1, e2 ) );
	bandwidth_in_MBs= 1e3f * max_iter * (mem_size * 2.0f) / 
	       	(elapsed_time_in_Ms * (float)(1 << 20));
	fprintf( stdout, "pinned_async gpu= %d size(MB)= %9u time(ms)= %.3f bandwidth(MB/s)= %.1f\n",
		d, mem_size/(1<<20), elapsed_time_in_Ms/(max_iter),
	       	bandwidth_in_MBs );
	fflush(stdout);

	if( check( h_data, 1e0f, nelem) == 0 )
		fprintf( stdout, "test FAILED\n" );

	// clean up memory
	CUDA_SAFE_CALL( cudaEventDestroy( e1 ) );
	CUDA_SAFE_CALL( cudaEventDestroy( e2 ) );
	CUDA_SAFE_CALL( cudaFreeHost( h_data ) );
	CUDA_SAFE_CALL( cudaFree( d_data ) );
	}

	cudaThreadExit();
}

