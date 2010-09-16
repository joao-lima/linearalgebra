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

#include <cutil_inline.h>

#include "add_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
	unsigned int mem_size= (1 << 26);
	float elapsed_time_in_Ms= 0;
	float bandwidth_in_MBs= 0;
	int i, max_iter= 10;
	float *h_data, *d_data;

	if( argc > 1 )
		mem_size =  (1 << atoi(argv[1]));

	unsigned int nelem= mem_size/sizeof(float);
	cudaSetDevice( 1 );
	/* CUDA flags:
	cudaHostAllocDefault, cudaHostAllocPortable, cudaHostAllocMapped,
	cudaHostAllocWriteCombined */
	unsigned int flags= cudaHostAllocWriteCombined;
	// allocate host memory for matrices A and B
	CUDA_SAFE_CALL( cudaHostAlloc( (void**)&h_data, mem_size, flags ) );
	for( i= 0; i < nelem; i++) h_data[i]= 1e0f;
	// allocate device memory
	CUDA_SAFE_CALL( cudaMalloc((void**)&d_data, mem_size) );
	cudaEvent_t e1, e2;
	cudaEventCreate( &e1 );
	cudaEventCreate( &e2 );
	// setup execution parameters
	dim3 threads( BLOCK_SIZE, 1 );
	dim3 grid( nelem / threads.x, 1 );

	CUDA_SAFE_CALL( cudaEventRecord( e1, 0 ) );
	for( i= 0; i < max_iter; i++ ){
		CUDA_SAFE_CALL( cudaMemcpy( d_data, h_data, mem_size,
				      cudaMemcpyHostToDevice) );
		add_one<<< grid, threads >>>( d_data );
		CUDA_SAFE_CALL( cudaMemcpy( h_data, d_data, mem_size,
				      cudaMemcpyDeviceToHost) );
	}
	CUDA_SAFE_CALL( cudaEventRecord( e2, 0 ) );
	CUDA_SAFE_CALL( cudaEventSynchronize( e2 ) );
	CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsed_time_in_Ms, e1, e2 ) );
	bandwidth_in_MBs= 1e3f * max_iter * (mem_size * 2.0f) / 
	       	(elapsed_time_in_Ms * (float)(1 << 20));
	fprintf( stdout, "pinned_wc size(MB)= %9u time(s)= %.3f bandwidth(MB/s)= %.1f\n",
		mem_size/(1<<20), elapsed_time_in_Ms/(1e3f*max_iter),
	       	bandwidth_in_MBs );

	if( check( h_data, 11e0f, nelem) == 1 )
		fprintf( stdout, "test OK\n" );
	else
		fprintf( stdout, "test FAILED\n" );

	// clean up memory
	CUDA_SAFE_CALL( cudaEventDestroy( e1 ) );
	CUDA_SAFE_CALL( cudaEventDestroy( e2 ) );
	CUDA_SAFE_CALL( cudaFreeHost( h_data ) );
	CUDA_SAFE_CALL( cudaFree( d_data ) );

	cudaThreadExit();
}

