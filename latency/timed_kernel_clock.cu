 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
 
#ifndef DEVICE
#define DEVICE 0
#endif

#include "cuda_safe.h"

#define NUM_SMS (24)
#define NUM_THREADS_PER_SM (384)
#define NUM_THREADS_PER_BLOCK (192)
#define NUM_BLOCKS ((NUM_THREADS_PER_SM / NUM_THREADS_PER_BLOCK) * NUM_SMS)
#define NUM_ITERATIONS 99999
 
// 128 MAD instructions
#define FMAD128(a, b) \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
 
__shared__ float result[NUM_THREADS_PER_BLOCK];
 
__global__ void gflops( unsigned int n, clock_t *timer )
{
	__threadfence();
	if( threadIdx.x == 0 )
		timer[0]= clock();

   float a = result[threadIdx.x];  // this ensures the mads don't get compiled out
   float b = 1.01f;
 
   for (int i = 0; i < n; i++)
   {
       FMAD128(a, b);
       FMAD128(a, b);
       FMAD128(a, b);
       FMAD128(a, b);
       FMAD128(a, b);
       FMAD128(a, b);
       FMAD128(a, b);
       FMAD128(a, b);
       FMAD128(a, b);
       FMAD128(a, b);
       FMAD128(a, b);
       FMAD128(a, b);
       FMAD128(a, b);
       FMAD128(a, b);
       FMAD128(a, b);
       FMAD128(a, b);
   }
   result[threadIdx.x] = a + b;

	__threadfence();
	if( threadIdx.x == 0 )
		timer[1]= clock();
}
 
int
main(int argc, char** argv)
{
	int max_work=1000;
	unsigned int mem_size;
	clock_t *d_timer, *h_timer;
        cudaDeviceProp deviceProp;
	float sys_time;
	float uclock;
	int i, max_iter= 100;
	struct timeval t0, t1;

	CUDA_SAFE_CALL( cudaSetDevice(DEVICE) );
	mem_size= sizeof(clock_t) * 2;
	h_timer= (clock_t *) malloc(mem_size);
	cudaMalloc( (void**)&d_timer, mem_size );
        cudaGetDeviceProperties(&deviceProp, DEVICE);

	gflops<<<1, 1>>>(max_work, d_timer);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	fprintf( stdout, "# clock(us) cpu(us) diff(us)\n" );
	fflush( stdout );

	for( i= 0; i < max_iter; i++ ) {
		CUDA_SAFE_CALL( cudaThreadSynchronize() );
		gettimeofday( &t0, 0 );
		gflops<<<1, 1>>>(max_work, d_timer);
		CUDA_SAFE_CALL( cudaThreadSynchronize() );
		gettimeofday( &t1, 0 );
		sys_time= (t1.tv_sec-t0.tv_sec)*1e6+(t1.tv_usec-t0.tv_usec);
		cudaMemcpy( h_timer, d_timer, mem_size, cudaMemcpyDeviceToHost);
		if( (h_timer[1]-h_timer[0]) > 0 ){
			uclock= (h_timer[1]-h_timer[0])/(deviceProp.clockRate*1e3f);
			fprintf( stdout, "%.2f %.2f %.2f\n", uclock*1e6,
					sys_time,
				sys_time-uclock*1e6 );
			fflush(stdout);
		}
	}
	cudaThreadExit();
}
