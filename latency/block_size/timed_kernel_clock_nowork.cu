 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
 
#ifndef DEVICE
#define DEVICE 0
#endif

#include "cuda_safe.h"

__global__ void gflops( unsigned int n, clock_t *timer )
{
	int idx= blockIdx.x * blockDim.x + threadIdx.x;
	__threadfence();
	if( idx == 0 )
		timer[0]= clock();

	__threadfence();
	if( idx == 0 )
		timer[1]= clock();
}
 
int
main(int argc, char** argv)
{
	int max_work=0;
	unsigned int mem_size;
	clock_t *d_timer, *h_timer;
        cudaDeviceProp deviceProp;
	float sys_time;
	float launch_time;
	float uclock;
	int i, j, max_iter= 1, t;
	struct timeval t0, t1, t2;
	unsigned int max_sm= 128;
	unsigned int max_thread= 1024;
	unsigned int sm, thread;

	CUDA_SAFE_CALL( cudaSetDevice(DEVICE) );
	mem_size= sizeof(clock_t) * 2;
	h_timer= (clock_t *) malloc(mem_size);
	cudaMalloc( (void**)&d_timer, mem_size );
        cudaGetDeviceProperties(&deviceProp, DEVICE);

	gflops<<<1, 1>>>(max_work, d_timer);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	fprintf( stdout, "# SMs threads clock(us) launch(us) cpu(us) diff(us)\n" );
	fflush( stdout );

	for( i = 1; i <= max_sm; i++ ) {
	for( j = 0; pow(2,j) <= max_thread; j++ ) {
		sm= i;
		thread= pow(2, j);
		gflops<<<sm, thread>>>(max_work, d_timer);
		CUDA_SAFE_THREAD_SYNC();
		CUDA_SAFE_CALL( cudaThreadSynchronize() );
	for( t= 0; t < max_iter; t++ ) {
		gettimeofday( &t0, 0 );
		gflops<<<sm, thread>>>(max_work, d_timer);
		CUDA_SAFE_THREAD_SYNC();
		gettimeofday( &t1, 0 );
		CUDA_SAFE_CALL( cudaThreadSynchronize() );
		gettimeofday( &t2, 0 );
		launch_time= (t1.tv_sec-t0.tv_sec)*1e6+(t1.tv_usec-t0.tv_usec);
		sys_time= (t2.tv_sec-t0.tv_sec)*1e6+(t2.tv_usec-t0.tv_usec);
		cudaMemcpy( h_timer, d_timer, mem_size, cudaMemcpyDeviceToHost);
		if( (h_timer[1]-h_timer[0]) > 0 ){
			uclock= (h_timer[1]-h_timer[0])/(deviceProp.clockRate*1e3f);
			fprintf( stdout, "%4d %4d %10.2f %10.2f %10.2f %10.2f\n",
					sm,
					thread,
					uclock*1e6,
					launch_time,
					sys_time,
				sys_time-(uclock*1e6) );
			fflush(stdout);
		}
	}
	}
	}
	cudaThreadExit();
}
