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
	char *h_in, *h_out, *d_in, *d_out;

	if( argc > 1 )
		mem_size =  (1 << atoi(argv[1]));

	/* CUDA device flags: cudaDeviceScheduleAuto, cudaDeviceScheduleSpin,
	   cudaDeviceScheduleYield, cudaDeviceBlockingSync, cudaDeviceMapHost
	   */
	unsigned int flags= cudaDeviceMapHost;
	CUDA_SAFE_CALL( cudaSetDeviceFlags( flags ) );
	cudaSetDevice( 1 );

	/* CUDA flags:
	cudaHostAllocDefault, cudaHostAllocPortable, cudaHostAllocMapped,
	cudaHostAllocWriteCombined */
	flags= cudaHostAllocMapped | cudaHostAllocWriteCombined;
	// allocate host memory for matrices A and B
	CUDA_SAFE_CALL( cudaHostAlloc( (void**)&h_in, mem_size, flags ) );
	CUDA_SAFE_CALL( cudaHostAlloc( (void**)&h_out, mem_size, flags ) );
	memset( h_in, 1, mem_size );
	// allocate device memory
	CUDA_SAFE_CALL( cudaHostGetDevicePointer((void**) &d_in, h_in, 0) );
	CUDA_SAFE_CALL( cudaHostGetDevicePointer((void**) &d_out, h_out, 0) );
	cudaEvent_t e1, e2;
	cudaEventCreate( &e1 );
	cudaEventCreate( &e2 );
	// setup execution parameters
	dim3 threads( BLOCK_SIZE, 1 );
	dim3 grid( mem_size / threads.x, 1 );

	CUDA_SAFE_CALL( cudaEventRecord( e1, 0 ) );
	for( i= 0; i < max_iter; i++ ){
		add_one<<< grid, threads >>>( d_out, (const char*)d_in );
		cutilCheckMsg("Kernel execution failed");
	}
	CUDA_SAFE_CALL( cudaEventRecord( e2, 0 ) );
	CUDA_SAFE_CALL( cudaEventSynchronize( e2 ) );
	CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsed_time_in_Ms, e1, e2 ) );
	bandwidth_in_MBs= 1e3f * max_iter * (mem_size * 2.0f) / 
	       	(elapsed_time_in_Ms * (float)(1 << 20));
	fprintf( stdout, "mapped_wc size= %9u time(s)= %.3f bandwidth(MB/s)= %.1f\n",
		mem_size, elapsed_time_in_Ms/(1e3f*max_iter),
	       	bandwidth_in_MBs );

	// clean up memory
	CUDA_SAFE_CALL( cudaEventDestroy( e1 ) );
	CUDA_SAFE_CALL( cudaEventDestroy( e2 ) );
	CUDA_SAFE_CALL( cudaFreeHost( h_in ) );
	CUDA_SAFE_CALL( cudaFreeHost( h_out ) );

	cudaThreadExit();
}

