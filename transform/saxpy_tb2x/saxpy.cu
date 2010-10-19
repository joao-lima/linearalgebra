
#include <iostream>
#include <algorithm>

#include "cuda_defs.h"
#include "tb.h"

#include "saxpy_kernel.cu"
#include "tb_kernel.cu"

void randomInit( float* data, int size )
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

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

// Y <- A * X + Y
void saxpy( float A, const unsigned int N, const unsigned int gpu_sm_count )
{
	float *d_x, *d_y;
	int max_iter= 1;
	float elapsed_time_in_Ms= 0;
	float bandwidth_in_MBs= 0;
	unsigned int n_block, n_beg;
	unsigned int sm;
	float *x, *y, *ref_y;
	cudaStream_t k_stream, mem_stream;

	fprintf( stdout, "saxpy N=%d gpu_sm_count=%d\n", N, gpu_sm_count );
	fflush( stdout );
	cudaEvent_t e1, e2;
	unsigned int mem_size= N * sizeof(float);

	// init device
	unsigned int flags= cudaDeviceMapHost;
	CUDA_SAFE_CALL( cudaSetDeviceFlags(flags) );
	CUDA_SAFE_CALL( cudaSetDevice( DEVICE ) );

	// alloc x and y pointers
	CUDA_SAFE_CALL( cudaHostAlloc((void**)&x, mem_size,
				cudaHostAllocDefault) );
	CUDA_SAFE_CALL( cudaHostAlloc((void**)&y, mem_size,
				cudaHostAllocDefault) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&d_x, mem_size) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&d_y, mem_size) );
	ref_y= (float*) malloc( mem_size );
	// alloc thread poll
	flags= cudaHostAllocMapped;
	void *h_tb, *d_tb;
	CUDA_SAFE_CALL( cudaHostAlloc(&h_tb,
		gpu_sm_count * sizeof(tb_t), flags) );
	CUDA_SAFE_CALL( cudaHostGetDevicePointer(&d_tb, h_tb, 0) );

	// setup execution parameters
	cudaEventCreate( &e1 );
	cudaEventCreate( &e2 );
	cudaStreamCreate(&k_stream);
	cudaStreamCreate(&mem_stream);

	CUDA_SAFE_CALL( cudaEventRecord( e1, 0 ) );
	for( sm= 0; sm < gpu_sm_count; sm++ ) {
		volatile tb_t *tb = &((volatile tb_t*)h_tb)[sm];
		tb_init( tb );
	}
	// init kernel
	tb_kernel<<< gpu_sm_count, 1, 0, k_stream >>>( (volatile tb_t*)d_tb, d_x, d_y, N, saxpy_gpu(A));

	// init first computation
	randomInit( x, N );
	randomInit( y, N );
	memcpy( ref_y, y, mem_size );
	CUDA_SAFE_CALL( cudaMemcpyAsync( d_x, x, N*sizeof(float),
			      cudaMemcpyHostToDevice, mem_stream) );
	CUDA_SAFE_CALL( cudaMemcpyAsync( d_y, y, N*sizeof(float),
			      cudaMemcpyHostToDevice, mem_stream) );
	cudaStreamSynchronize(mem_stream);
	n_block= N / gpu_sm_count;
	n_beg= 0;
	for( sm= 0; sm < gpu_sm_count; sm++ ) {
		volatile tb_t* const tb = &((volatile tb_t*)h_tb)[sm];
		tb_post( tb, n_beg, n_block );
		n_beg += n_block;
	}
	fprintf( stdout, "saxpy POST done\n" );
	fflush( stdout );
	for( sm= 0; sm < gpu_sm_count; sm++ ) {
		volatile tb_t* const tb = &((volatile tb_t*)h_tb)[sm];
		tb_wait( tb );
	}
	fprintf( stdout, "saxpy WAIT done\n" );
	fflush( stdout );
	CUDA_SAFE_CALL( cudaMemcpyAsync( y, d_y, N*sizeof(float),
			      cudaMemcpyDeviceToHost, mem_stream) );
	cudaStreamSynchronize(mem_stream);
	check( 2.0, x, y, ref_y, N );

	// init second computation
	randomInit( x, N );
	randomInit( y, N );
	memcpy( ref_y, y, mem_size );
	CUDA_SAFE_CALL( cudaMemcpyAsync( d_x, x, N*sizeof(float),
			      cudaMemcpyHostToDevice, mem_stream) );
	CUDA_SAFE_CALL( cudaMemcpyAsync( d_y, y, N*sizeof(float),
			      cudaMemcpyHostToDevice, mem_stream) );
	cudaStreamSynchronize(mem_stream);
	fprintf( stdout, "saxpy POST AGAIN\n" );
	fflush( stdout );
	n_beg= 0;
	for( sm= 0; sm < gpu_sm_count; sm++ ) {
		volatile tb_t* const tb = &((volatile tb_t*)h_tb)[sm];
		tb_post( tb, n_beg, n_block );
		n_beg += n_block;
	}
	fprintf( stdout, "saxpy POST AGAIN done\n" );
	fflush( stdout );
	for( sm= 0; sm < gpu_sm_count; sm++ ) {
		volatile tb_t* const tb = &((volatile tb_t*)h_tb)[sm];
		tb_wait( tb );
	}
	CUDA_SAFE_CALL( cudaMemcpyAsync( y, d_y, N*sizeof(float),
			      cudaMemcpyDeviceToHost, mem_stream) );
	cudaStreamSynchronize(mem_stream);
	check( 2.0, x, y, ref_y, N );
	fprintf( stdout, "saxpy WAIT AGAIN done\n" );
	fflush( stdout );

	// finish kernel
	for( sm= 0; sm < gpu_sm_count; sm++ ) {
		volatile tb_t* const tb = &((volatile tb_t*)h_tb)[sm];
		tb_finish( tb );
	}
	fprintf( stdout, "saxpy finish done\n" );
	fflush( stdout );
	CUDA_SAFE_CALL( cudaEventRecord( e2, 0 ) );
	CUDA_SAFE_CALL( cudaEventSynchronize( e2 ) );

	CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsed_time_in_Ms, e1, e2 ) );
	bandwidth_in_MBs= 1e3f * max_iter * (mem_size * 3.0f) / 
	       	(elapsed_time_in_Ms * (float)(1 << 20));
	fprintf( stdout, "saxpy n=%d size(MB)= %9u time(ms)= %.3f bandwidth(MB/s)= %.1f\n",
		N, mem_size/(1<<20), elapsed_time_in_Ms/max_iter,
	       	bandwidth_in_MBs );
	

	// clean up memory
	cudaStreamDestroy(mem_stream);
	cudaStreamDestroy(k_stream);
	CUDA_SAFE_CALL( cudaEventDestroy( e1 ) );
	CUDA_SAFE_CALL( cudaEventDestroy( e2 ) );
	CUDA_SAFE_CALL( cudaFree(d_x) );
	CUDA_SAFE_CALL( cudaFree(d_y) );
	CUDA_SAFE_CALL( cudaFreeHost(h_tb) );
	cudaFreeHost( x );
	cudaFreeHost( y );
	free( ref_y );
}

int main( int argc, char *argv[] )
{
	unsigned int mem_size= (1 << 25);
	unsigned int gpu_sm_count= 1;

	if( argc > 1 )
		mem_size =  (1 << atoi(argv[1]));
	if( argc > 2 )
		gpu_sm_count = atoi(argv[2]);
	unsigned int nelem= mem_size/sizeof(float);

#if 0
	x= (float*) malloc( mem_size );
	y= (float*) malloc( mem_size );
#endif
	saxpy( 2.0, nelem, gpu_sm_count );
	cudaThreadExit();
}

