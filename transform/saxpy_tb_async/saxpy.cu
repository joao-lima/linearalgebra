
#include <iostream>
#include <algorithm>

#include "cuda_defs.h"
#include "tb.h"

#include "saxpy_kernel.cu"
#include "tb_kernel.cu"

void check( float A, float *x, float *y, float *ref_y, unsigned int N )
{
	int result;

	std::transform( x, x+N, ref_y, ref_y, saxpy_gpu(A) );
	result= compareL2fe( ref_y, y, N, 1e-6f );
	if( result == 0 ) {
		fprintf( stdout, "ERROR\n" );
		fprintf( stdout, "%f %f\n", y[0], ref_y[0] );
		fprintf( stdout, "%f %f\n", y[44], ref_y[44] );
		fprintf( stdout, "%f %f\n", y[N-1], ref_y[N-1] );
	}
}

void saxpy( float A, float *x, float *y, const unsigned int N,
	const unsigned int gpu_sm_count )
{
	float *d_x, *d_y;
	int max_iter= 1;
	float elapsed_time_in_Ms= 0;
	float bandwidth_in_MBs= 0;
	unsigned int n_block, n_beg;
	unsigned int sm;
	cudaStream_t kernel_stream, mem_stream;

	fprintf( stdout, "saxpy N=%d gpu_sm_count=%d\n", N, gpu_sm_count );
	fflush( stdout );
	cudaEvent_t e1, e2;
	unsigned int mem_size= N * sizeof(float);
	//unsigned int flags= cudaDeviceMapHost;
	//CUDA_SAFE_CALL( cudaSetDeviceFlags(flags) );
	CUDA_SAFE_CALL( cudaSetDevice( DEVICE ) );

	// Y <- A * X + Y
	// setup execution parameters
	CUDA_SAFE_CALL( cudaEventCreate( &e1 ) );
	CUDA_SAFE_CALL( cudaEventCreate( &e2 ) );
	CUDA_SAFE_CALL( cudaStreamCreate( &kernel_stream ) );
	CUDA_SAFE_CALL( cudaStreamCreate( &mem_stream ) );

	CUDA_SAFE_CALL( cudaMalloc((void**)&d_x, mem_size) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&d_y, mem_size) );
	CUDA_SAFE_CALL( cudaMemcpy( d_x, x, N*sizeof(float),
			      cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy( d_y, y, N*sizeof(float),
			      cudaMemcpyHostToDevice) );

	// alloc thread table
	unsigned int flags= cudaHostAllocDefault;
	void *h_tb, *d_tb;
	unsigned int memsize_tb= gpu_sm_count * sizeof(tb_t);
	CUDA_SAFE_CALL( cudaHostAlloc(&h_tb, memsize_tb, flags) );
	CUDA_SAFE_CALL( cudaMalloc(&d_tb, memsize_tb ) );

	CUDA_SAFE_CALL( cudaEventRecord( e1, 0 ) );
	for( sm= 0; sm < gpu_sm_count; sm++ ) {
		volatile tb_t *tb = &((volatile tb_t*)h_tb)[sm];
		tb_init( tb );
	}
	CUDA_SAFE_CALL( cudaMemcpy(d_tb, h_tb, memsize_tb,
				cudaMemcpyHostToDevice) );
	tb_kernel<<< gpu_sm_count, 1, 0, kernel_stream >>>( (volatile tb_t*)d_tb, d_x, d_y, N, saxpy_gpu(A));
	n_block= N / gpu_sm_count;
	n_beg= 0;
	for( sm= 0; sm < gpu_sm_count; sm++ ) {
		volatile tb_t* const tb = &((volatile tb_t*)h_tb)[sm];
		tb_post( tb, n_beg, n_block );
		n_beg += n_block;
	}
	CUDA_SAFE_CALL( cudaMemcpyAsync(d_tb, h_tb, memsize_tb,
				cudaMemcpyHostToDevice, mem_stream) );
	CUDA_SAFE_CALL( cudaStreamSynchronize(mem_stream) );
	fprintf( stdout, "saxpy post done\n" );
	fflush( stdout );
#if 1
tb_loop:
	CUDA_SAFE_CALL( cudaMemcpyAsync(h_tb, d_tb, memsize_tb,
				cudaMemcpyDeviceToHost, mem_stream) );
	CUDA_SAFE_CALL( cudaStreamSynchronize(mem_stream) );
	for( sm= 0; sm < gpu_sm_count; sm++ ) {
		volatile tb_t* const tb = &((volatile tb_t*)h_tb)[sm];
		if( tb->status != TB_READY )
			goto tb_loop;
	}
	fprintf( stdout, "saxpy wait done\n" );
	fflush( stdout );
#endif
	for( sm= 0; sm < gpu_sm_count; sm++ ) {
		volatile tb_t* const tb = &((volatile tb_t*)h_tb)[sm];
		tb_finish( tb );
	}
	CUDA_SAFE_CALL( cudaMemcpyAsync(d_tb, h_tb, memsize_tb,
				cudaMemcpyHostToDevice, mem_stream) );
	CUDA_SAFE_CALL( cudaStreamSynchronize(mem_stream) );
	fprintf( stdout, "saxpy finish done\n" );
	fflush( stdout );
	CUDA_SAFE_CALL( cudaMemcpy( y, d_y, N*sizeof(float),
			      cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaStreamSynchronize(mem_stream) );
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
	CUDA_SAFE_CALL( cudaFree(d_x) );
	CUDA_SAFE_CALL( cudaFree(d_y) );
	CUDA_SAFE_CALL( cudaFreeHost(h_tb) );
}

void randomInit( float* data, int size )
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

int main( int argc, char *argv[] )
{
	unsigned int mem_size= (1 << 25);
	float *x, *y, *ref_y;
	unsigned int gpu_sm_count= 1;

	if( argc > 1 )
		mem_size =  (1 << atoi(argv[1]));
	if( argc > 2 )
		gpu_sm_count = atoi(argv[2]);
	unsigned int nelem= mem_size/sizeof(float);

	x= (float*) malloc( mem_size );
	y= (float*) malloc( mem_size );
	ref_y= (float*) malloc( mem_size );
	randomInit( x, nelem );
	randomInit( y, nelem );
	memcpy( ref_y, y, mem_size );
	saxpy( 2.0, x, y, nelem, gpu_sm_count );
	check( 2.0, x, y, ref_y, nelem );
	free( x );
	free( y );
	free( ref_y );
	cudaThreadExit();
}

