/*
   This sample is intended to measure the peak computation rate of the GPU in GFLOPs
   (giga floating point operations per second).
 
   It executes a large number of multiply-add operations, writing the results to
   shared memory. The loop is unrolled for maximum performance.
 
   Depending on the compiler and hardware it might not take advantage of all the
   computational resources of the GPU, so treat the results produced by this code
   with some caution.
*/
 
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
	struct timeval e0, e1, e2, e3, ek1, ek2;
	cudaEvent_t ev1, ev2;
	float t0, t1, t2, ta, tk;
	int min_iter= 2, max_iter=10000;
	int i, j, nmax=100;
	float t_temp, elapsed_time= 0.0f;

	CUDA_SAFE_CALL( cudaSetDevice(DEVICE) );
	h= (unsigned int *) malloc(sizeof(unsigned int));
	*h= 1;
	cudaMalloc((void**)&d, sizeof(unsigned int));
	cudaEventCreate( &ev1 );
	cudaEventCreate( &ev2 );

	cudaMemcpy( d, h, sizeof(unsigned int), cudaMemcpyHostToDevice );
	gflops<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(min_iter);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	for( i= min_iter; i <= max_iter; i= i * 2 ) {
		t0= t1= t2= ta= elapsed_time= 0.0f;
		// ops
		gflops<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(i);
		cudaThreadSynchronize();

		//gettimeofday( &ek1, 0 );
		cudaEventRecord( ev1, 0 );
		for( j= 0; j < nmax; j++ ){
		gflops<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(i);
		cudaThreadSynchronize();
		}
		//gettimeofday( &ek2, 0 );
		cudaEventRecord( ev2, 0 );
		cudaEventSynchronize( ev2 );
		cudaEventElapsedTime( &t_temp, ev1, ev2 );
		tk= (t_temp*1e3)/nmax;
		//tk= ((ek2.tv_sec-ek1.tv_sec)*1e6+(ek2.tv_usec-ek1.tv_usec))/nmax;

		//fprintf( stdout, "kernel iter= %d time= %f\n", i, tk );
		//fflush(stdout);

		for( j= 0; j < nmax; j++ ){
		gettimeofday( &e0, 0 );
		cudaEventRecord( ev1, 0 );
		gflops<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(i);
		cudaEventRecord( ev2, 0 );
		gettimeofday( &e1, 0 );
		cudaMemcpy( d, h, sizeof(unsigned int), cudaMemcpyHostToDevice );
		gettimeofday( &e2, 0 );
		cudaThreadSynchronize();
		cudaEventSynchronize( ev2 );
		gettimeofday( &e3, 0 );
		t0+= (e1.tv_sec-e0.tv_sec)*1e6+(e1.tv_usec-e0.tv_usec);
		t1+= (e2.tv_sec-e1.tv_sec)*1e6+(e2.tv_usec-e1.tv_usec);
		t2+= (e3.tv_sec-e2.tv_sec)*1e6+(e3.tv_usec-e2.tv_usec);
		ta+= (e3.tv_sec-e0.tv_sec)*1e6+(e3.tv_usec-e0.tv_usec);
		cudaEventElapsedTime( &t_temp, ev1, ev2 );
		elapsed_time += t_temp;
		}
	t0= t0/nmax; 
	t1= t1/nmax;
	t2= t2/nmax;
	ta= ta/nmax;
	elapsed_time = elapsed_time/nmax;

	//fprintf( stdout, "i= %d total= %f t0= %f t1= %f t2= %f\n", i, 
	//		ta, t0, t1, t2 );
	fprintf( stdout, "res(%4d) tk= %7.2f ta= %7.2f t1= %7.2f\n",
			i, tk, ta, t0, t1 );
		fflush(stdout);
	}
	cudaEventDestroy( ev1 );
	cudaEventDestroy( ev2 );
	cudaThreadExit();
}
