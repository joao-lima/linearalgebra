 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
 
#ifndef DEVICE
#define DEVICE 0
#endif

#include "cuda_safe.h"
#include "kernel_clock_timed.cu"
 
int
main(int argc, char** argv)
{
	clock_t *d_timer, *h_timer;
	struct timeval t0, t1, t2;
	float time1_0, time2_0;
	float k_time;
	int max_work=1000;
	double max_time= 100*1e3;
	unsigned long mem_size, mem_size_clock, shared_mem_size;
	int i, nmax=100;
	unsigned int sm= 30, thread= 128;
	cudaStream_t stream1;
        cudaDeviceProp deviceProp;
	float *d_data;
	unsigned int mem_size_result;
	unsigned int N;
	unsigned int offset= 0;

	if( argc > 1 )
		max_work= atol( argv[1] );

	CUDA_SAFE_CALL( cudaSetDevice(DEVICE) );
	shared_mem_size= 0;

	mem_size_result= sm * thread * sizeof(float);
	N= sm * thread;
	cudaMalloc( (void**)&d_data, mem_size_result );

	cudaStreamCreate( &stream1 );

	mem_size_clock= sizeof(clock_t) * 2;
	CUDA_SAFE_CALL( cudaHostAlloc( &h_timer, mem_size_clock,
				cudaHostAllocDefault ) );
	cudaMalloc( (void**)&d_timer, mem_size_clock );
        cudaGetDeviceProperties(&deviceProp, DEVICE);

	gflops_heavy<<<sm, thread, shared_mem_size, stream1>>>(max_work,
			d_data, N, offset, d_timer);
	CUDA_SAFE_THREAD_SYNC();
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	/* search for some work near to 100 ms */
	i= max_work;
	do {
		do {
		gflops_heavy<<<sm, thread, shared_mem_size, stream1>>>(i,
			d_data, N, offset, d_timer);
		CUDA_SAFE_CALL( cudaStreamSynchronize(stream1) );
		CUDA_SAFE_CALL( cudaMemcpy( h_timer, d_timer, mem_size_clock,
			cudaMemcpyDeviceToHost) );
		k_time= 1e6*((h_timer[1]-h_timer[0])/(deviceProp.clockRate*1e3f));
		} while( k_time < 0 );
		i+=10;
	} while( k_time < max_time );
	max_work= i-10;

	fprintf( stdout, "# max_work=%d time=%.2f \n", max_work, k_time );
	fprintf( stdout, "# size(B) t1-t0 t2-t0 time_kernel\n" );
	fflush(stdout);

	time1_0= time2_0= k_time= 0;

	for( i= 0; i < nmax; i++ ){
		gettimeofday( &t0, 0 );
		gflops_heavy<<<sm, thread, shared_mem_size, stream1>>>(max_work,
			       d_data, N, offset, d_timer);
		CUDA_SAFE_THREAD_SYNC();
		gettimeofday( &t1, 0 );
		CUDA_SAFE_CALL( cudaStreamSynchronize(stream1) );
		gettimeofday( &t2, 0 );
		time1_0= (t1.tv_sec-t0.tv_sec)*1e6+(t1.tv_usec-t0.tv_usec);
		time2_0= (t2.tv_sec-t0.tv_sec)*1e6+(t2.tv_usec-t0.tv_usec);
		CUDA_SAFE_CALL( cudaMemcpy( h_timer, d_timer, mem_size_clock,
				cudaMemcpyDeviceToHost) );
		k_time= 1e6*((h_timer[1]-h_timer[0])/(deviceProp.clockRate*1e3f));
		if(  k_time > 0 ){
			fprintf( stdout, "%10u %10.2f %10.2f %10.2f\n",
				mem_size,
				time1_0, time2_0, k_time );
		} else {
			fprintf( stdout, "# %10u %10.2f %10.2f %10.2f\n",
				mem_size,
				time1_0, time2_0, k_time );
		}
	}
	fflush(stdout);
	cudaFree( d_data );
	cudaFreeHost( h_timer );
	cudaFree( d_timer );
	cudaThreadExit();
}
