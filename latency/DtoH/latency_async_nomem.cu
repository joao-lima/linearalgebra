 
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
	unsigned int *h, *d;
	struct timeval ek0, ek1, t0, t1, t2, t3;
	float time1_0, time2_0, time3_0, tk;
	float k_time;
	int max_work=2800;
	unsigned int mem_max= 30;
	unsigned long mem_size, mem_size_clock, shared_mem_size;
	int i, j, nmax=100;
	unsigned int sm= 30, thread= 128;
	cudaStream_t stream1, stream2;
        cudaDeviceProp deviceProp;
	float *d_data;
	unsigned int mem_size_result;
	unsigned int N;
	unsigned int offset= 0;
	unsigned int flops;

	if( argc > 1 )
		mem_max= atoi(argv[1]);
	if( argc > 2 )
		max_work = atoi(argv[2]);

	CUDA_SAFE_CALL( cudaSetDevice(DEVICE) );
	mem_size= powl(2, mem_max);
	shared_mem_size= 0;
	CUDA_SAFE_CALL( cudaHostAlloc( &h, mem_size, cudaHostAllocDefault ) );
	for( i= 0; i < mem_size/sizeof(unsigned int); i++ ) h[i]= 1;
	cudaMalloc( (void**)&d, mem_size );

	flops = 128 * 2 * 16 * max_work * sm * thread;
	mem_size_result= sm * thread * sizeof(float);
	N= sm * thread;
	cudaMalloc( (void**)&d_data, mem_size_result );

	cudaStreamCreate( &stream1 );
	cudaStreamCreate( &stream2 );

	mem_size_clock= sizeof(clock_t) * 2;
	CUDA_SAFE_CALL( cudaHostAlloc( &h_timer, mem_size_clock,
				cudaHostAllocDefault ) );
	cudaMalloc( (void**)&d_timer, mem_size_clock );
        cudaGetDeviceProperties(&deviceProp, DEVICE);

	cudaMemcpy( d, h, mem_size, cudaMemcpyHostToDevice );
	gflops_light<<<sm, thread, shared_mem_size, 0>>>(max_work, d_data, N,
			offset, d_timer);
	CUDA_SAFE_THREAD_SYNC();
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	gettimeofday( &ek0, 0 );
	for( j= 0; j < nmax; j++ ){
	gflops_light<<<sm, thread, shared_mem_size, 0>>>(max_work, d_data, 
			N, offset, d_timer);
	CUDA_SAFE_THREAD_SYNC();
	}
	cudaThreadSynchronize();
	gettimeofday( &ek1, 0 );
	tk= ((ek1.tv_sec-ek0.tv_sec)*1e6+(ek1.tv_usec-ek0.tv_usec))/nmax;
	fprintf( stdout, "# kernel time= %f, flops=%d\n", tk, flops );
	fprintf( stdout, "# size(B) t1-t0 t2-t0 t3-t0 time_kernel\n" );
	fflush(stdout);

	unsigned int x;
	for( x= 0; x <= 2; x++ ) {
	for( i= 64; i <= 1024; i=i+64 ) {
		mem_size= i*powl(1024,x);

		time1_0= time2_0= time3_0= k_time= 0;

		cudaMemcpy( d, h, mem_size, cudaMemcpyDeviceToHost);

		for( j= 0; j < nmax; j++ ){
		gettimeofday( &t0, 0 );
		gflops_light<<<sm, thread, shared_mem_size, stream1>>>(max_work,
			       d_data, N, offset, d_timer);
		CUDA_SAFE_THREAD_SYNC();
		gettimeofday( &t1, 0 );
		CUDA_SAFE_CALL( cudaMemcpyAsync( d, h, mem_size,
					cudaMemcpyDeviceToHost, stream2 ) );
		CUDA_SAFE_CALL( cudaStreamSynchronize(stream2) );
		gettimeofday( &t2, 0 );
		CUDA_SAFE_CALL( cudaStreamSynchronize(stream1) );
		gettimeofday( &t3, 0 );
		time1_0= (t1.tv_sec-t0.tv_sec)*1e6+(t1.tv_usec-t0.tv_usec);
		time2_0= (t2.tv_sec-t0.tv_sec)*1e6+(t2.tv_usec-t0.tv_usec);
		time3_0= (t3.tv_sec-t0.tv_sec)*1e6+(t3.tv_usec-t0.tv_usec);
		CUDA_SAFE_CALL( cudaMemcpy( h_timer, d_timer, mem_size_clock,
			       	cudaMemcpyDeviceToHost) );
		k_time= 1e6*((h_timer[1]-h_timer[0])/(deviceProp.clockRate*1e3f));
		if( k_time > 0 ){
			fprintf( stdout, "%10u %10.2f %10.2f %10.2f %10.2f\n",
				mem_size,
				time1_0, time2_0, time3_0, k_time );
		} else {
			fprintf( stdout, "# %10u %10.2f %10.2f %10.2f %10.2f\n",
				mem_size,
				time1_0, time2_0, time3_0, k_time );
		}
		fflush(stdout);
		}
	}
	}
	cudaFreeHost( h );
	cudaFree( d );
	cudaFree( d_data );
	cudaFreeHost( h_timer );
	cudaFree( d_timer );
	cudaThreadExit();
}
