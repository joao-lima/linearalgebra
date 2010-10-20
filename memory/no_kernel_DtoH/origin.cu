
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
	unsigned int i, max_iter= 10;
	float *h_data, *d_data;

	if( argc > 1 )
		mem_size =  (1 << atoi(argv[1]));
	unsigned int nelem= mem_size/sizeof(float);

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	for( int d= 0; d < deviceCount; d++ ) {
	cudaSetDevice( d );
	// allocate host memory for matrices A and B
	h_data= (float*)malloc( mem_size );
	for( i= 0; i < nelem; i++) h_data[i]= 1e0f;
	// allocate device memory
	CUDA_SAFE_CALL( cudaMalloc((void**)&d_data, mem_size) );
	cudaEvent_t e1, e2;
	cudaEventCreate( &e1 );
	cudaEventCreate( &e2 );

	CUDA_SAFE_CALL( cudaEventRecord( e1, 0 ) );
	for( i= 0; i < max_iter; i++ ){
		CUDA_SAFE_CALL( cudaMemcpy( h_data, d_data, mem_size,
				      cudaMemcpyDeviceToHost) );
	}
	CUDA_SAFE_CALL( cudaEventRecord( e2, 0 ) );
	CUDA_SAFE_CALL( cudaEventSynchronize( e2 ) );
	CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsed_time_in_Ms, e1, e2 ) );
	bandwidth_in_MBs= 1e3f * max_iter * mem_size / 
	       	(elapsed_time_in_Ms * (float)(1 << 20));
	fprintf( stdout, "naive gpu= %d size(MB)= %9u time(ms)= %.3f bandwidth(MB/s)= %.1f\n",
		d, mem_size/(1<<20), elapsed_time_in_Ms/(max_iter),
	       	bandwidth_in_MBs );

	if( check( h_data, 1e0f, nelem) == 0 )
		fprintf( stdout, "test FAILED\n" );

	// clean up memory
	CUDA_SAFE_CALL( cudaEventDestroy( e1 ) );
	CUDA_SAFE_CALL( cudaEventDestroy( e2 ) );
	free( h_data );
	CUDA_SAFE_CALL( cudaFree( d_data ) );
	}

	cudaThreadExit();
}

