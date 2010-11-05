 
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
 
__global__ void gflops(unsigned int n)
{
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
}
 
int
main(int argc, char** argv)
{
	unsigned int *h, *d;
	struct timeval ek0, ek1, t0, t1, t2, t3;
	float time1_0, time2_0, time3_0, tk;
	int max_iter=1;
	// from (1<<mem_min) to (1<<mem_max)
	unsigned int mem_min= 1, mem_max= 30;
	unsigned int mem_size;
	int i, j, nmax=100;
	cudaStream_t stream1, stream2;

	CUDA_SAFE_CALL( cudaSetDevice(DEVICE) );
	mem_size= (1<<mem_max);
	CUDA_SAFE_CALL( cudaHostAlloc( &h, mem_size, cudaHostAllocDefault ) );
	//h= (unsigned int *) malloc(mem_size);
	for( i= 0; i < mem_size/sizeof(unsigned int); i++ ) h[i]= 1;
	cudaMalloc( (void**)&d, mem_size );
	cudaStreamCreate( &stream1 );
	cudaStreamCreate( &stream2 );

	cudaMemcpy( d, h, mem_size, cudaMemcpyHostToDevice );
	gflops<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(max_iter);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	gettimeofday( &ek0, 0 );
	for( j= 0; j < nmax; j++ ){
	gflops<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(max_iter);
	}
	cudaThreadSynchronize();
	gettimeofday( &ek1, 0 );
	tk= ((ek1.tv_sec-ek0.tv_sec)*1e6+(ek1.tv_usec-ek0.tv_usec))/nmax;
	fprintf( stdout, "kernel time= %f\n", tk );
	fflush(stdout);

	for( i= mem_min; i <= mem_max; i++ ) {
		mem_size= (1<<i);
		for( j= 0; j < 10; j++ )
			cudaMemcpy( d, h, mem_size, cudaMemcpyHostToDevice );

		for( j= 0; j < nmax; j++ ){
		gettimeofday( &t0, 0 );
		gflops<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, stream1>>>(max_iter);
		gettimeofday( &t1, 0 );
		CUDA_SAFE_CALL( cudaMemcpyAsync( d, h, mem_size, cudaMemcpyHostToDevice, stream2 ) );
		CUDA_SAFE_CALL( cudaStreamSynchronize(stream2) );
		gettimeofday( &t2, 0 );
		CUDA_SAFE_CALL( cudaStreamSynchronize(stream1) );
		//CUDA_SAFE_CALL( cudaThreadSynchronize() );
		gettimeofday( &t3, 0 );
		time1_0+= (t1.tv_sec-t0.tv_sec)*1e6+(t1.tv_usec-t0.tv_usec);
		time2_0+= (t2.tv_sec-t0.tv_sec)*1e6+(t2.tv_usec-t0.tv_usec);
		time3_0+= (t3.tv_sec-t0.tv_sec)*1e6+(t3.tv_usec-t0.tv_usec);
		}
		time1_0= time1_0/nmax;
		time2_0= time2_0/nmax;
		time3_0= time3_0/nmax;
		fprintf( stdout, "%10u %10.2f %10.2f %10.2f\n", mem_size,
			time1_0, time2_0, time3_0 );
		fflush(stdout);
	}
	cudaFreeHost( h );
	cudaFree( d );
	cudaThreadExit();
}
