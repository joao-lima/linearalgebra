 
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
	unsigned int *h, *d;
	struct timeval ek0, ek1, t0, t1, t2, t3;
	float time1_0, time2_0, time3_0, tk;
	int max_work=1;
	unsigned int mem_min= 0, mem_max= 30;
	float k_time;
	int n_k_time;
	unsigned long mem_size, mem_size_clock, shared_mem_size;
	int i, j, nmax=100;
        cudaDeviceProp deviceProp;
	unsigned int sm= 1, thread= 1;
	clock_t *d_timer, *h_timer;
	float kernel_work= 1.0f;

	if( argc > 1 )
		mem_max= atoi(argv[1]);

	CUDA_SAFE_CALL( cudaSetDevice(DEVICE) );
	mem_size= powl(2, mem_max);
	shared_mem_size= thread * sizeof(float);
	h= (unsigned int *) malloc(mem_size);
	for( i= 0; i < mem_size/sizeof(unsigned int); i++ ) h[i]= 1;
	cudaMalloc( (void**)&d, mem_size );

	mem_size_clock= sizeof(clock_t) * 2;
	h_timer= (clock_t *) malloc(mem_size_clock);
	cudaMalloc( (void**)&d_timer, mem_size_clock );
        cudaGetDeviceProperties(&deviceProp, DEVICE);

	cudaMemcpy( d, h, mem_size, cudaMemcpyHostToDevice );
	gflops<<<sm, thread, shared_mem_size, 0>>>(max_work, kernel_work, d_timer);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	gettimeofday( &ek0, 0 );
	for( j= 0; j < nmax; j++ ){
	gflops<<<sm, thread, shared_mem_size, 0>>>(max_work, kernel_work, d_timer);
	}
	cudaThreadSynchronize();
	gettimeofday( &ek1, 0 );
	tk= ((ek1.tv_sec-ek0.tv_sec)*1e6+(ek1.tv_usec-ek0.tv_usec))/nmax;
	fprintf( stdout, "# kernel time= %f\n", tk );
	fprintf( stdout, "# size(B) t1-t0 t2-t0 t3-t0 time_kernel\n" );
	fflush(stdout);

	for( i= mem_min; i <= mem_max; i++ ) {
		mem_size= powl(2, i);
		time1_0= time2_0= time3_0= k_time= 0;
		n_k_time= 0;
		for( j= 0; j < 2; j++ )
			cudaMemcpy( d, h, mem_size, cudaMemcpyHostToDevice );

		for( j= 0; j < nmax; j++ ){
		gettimeofday( &t0, 0 );
		gflops<<<sm, thread, shared_mem_size, 0>>>(max_work, kernel_work, d_timer);
		gettimeofday( &t1, 0 );
		cudaMemcpy( d, h, mem_size, cudaMemcpyHostToDevice );
		gettimeofday( &t2, 0 );
		cudaThreadSynchronize();
		gettimeofday( &t3, 0 );
		time1_0+= (t1.tv_sec-t0.tv_sec)*1e6+(t1.tv_usec-t0.tv_usec);
		time2_0+= (t2.tv_sec-t0.tv_sec)*1e6+(t2.tv_usec-t0.tv_usec);
		time3_0+= (t3.tv_sec-t0.tv_sec)*1e6+(t3.tv_usec-t0.tv_usec);
		CUDA_SAFE_CALL( cudaMemcpy( h_timer, d_timer, mem_size_clock,
			       	cudaMemcpyDeviceToHost) );
		if( (h_timer[1]-h_timer[0]) > 0 ){
			k_time+= (h_timer[1]-h_timer[0])/(deviceProp.clockRate*1e3f);
			n_k_time++;
		}
		}
		time1_0= time1_0/nmax;
		time2_0= time2_0/nmax;
		time3_0= time3_0/nmax;
		k_time= (k_time / n_k_time) * 1e6;
		fprintf( stdout, "%10u %10.2f %10.2f %10.2f %10.2f\n",
			mem_size,
			time1_0, time2_0, time3_0, k_time );
		fflush(stdout);
	}
	free( h );
	cudaFree( d );
	free( h_timer );
	cudaFree( d_timer );
	cudaThreadExit();
}
