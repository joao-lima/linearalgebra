
#include <iostream>

#include "cuda_defs.h"

#include "saxpy_kernel.cu"

void saxpy( float A, float *x, float *y, unsigned int N )
{
	float *d_x, *d_y;
	int i, max_iter= 10;
	cudaEvent_t e1, e2;
	dim3 threads( BLOCK_SIZE, 1 );
	dim3 grid( 128, 1);
	float elapsed_time_in_Ms= 0;
	float bandwidth_in_MBs= 0;
	unsigned int mem_size= N * sizeof(float);

	// Y <- A * X + Y
	// setup execution parameters
	cudaEventCreate( &e1 );
	cudaEventCreate( &e2 );
	CUDA_SAFE_CALL( cudaMalloc((void**)&d_x, mem_size) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&d_y, mem_size) );

	CUDA_SAFE_CALL( cudaEventRecord( e1, 0 ) );
	for( i= 0; i < max_iter; i++ ){
		CUDA_SAFE_CALL( cudaMemcpy( d_x, x, N*sizeof(float),
				      cudaMemcpyHostToDevice) );
		CUDA_SAFE_CALL( cudaMemcpy( d_y, y, N*sizeof(float),
				      cudaMemcpyHostToDevice) );
		saxpy_kernel<<< grid, threads >>>( d_x, d_y, N,
			saxpy_gpu(A) );
		CUDA_SAFE_CALL( cudaMemcpy( y, d_y, N*sizeof(float),
				      cudaMemcpyDeviceToHost) );
	}
	CUDA_SAFE_CALL( cudaEventRecord( e2, 0 ) );
	CUDA_SAFE_CALL( cudaEventSynchronize( e2 ) );

	CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsed_time_in_Ms, e1, e2 ) );
	bandwidth_in_MBs= 1e3f * max_iter * (mem_size * 3.0f) / 
	       	(elapsed_time_in_Ms * (float)(1 << 20));
	fprintf( stdout, "saxpy n=%d size(MB)= %9u time(ms)= %.3f bandwidth(MB/s)= %.1f\n",
		N, mem_size/(1<<20), elapsed_time_in_Ms/max_iter,
	       	bandwidth_in_MBs );

	// clean up memory
	CUDA_SAFE_CALL( cudaEventDestroy( e1 ) );
	CUDA_SAFE_CALL( cudaEventDestroy( e2 ) );
}

void randomInit( float* data, int size )
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

int main( int argc, char *argv[] )
{
	unsigned int mem_size= (1 << 25);
	float *x, *y;

	if( argc > 1 )
		mem_size =  (1 << atoi(argv[1]));
	unsigned int nelem= mem_size/sizeof(float);

	x= (float*) malloc( mem_size );
	y= (float*) malloc( mem_size );
	randomInit( x, nelem );
	randomInit( y, nelem );
	cudaSetDevice( DEVICE );
	saxpy( 2.0, x, y, nelem );
	free( x );
	free( y );
	cudaThreadExit();
}

