
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#include <cuda.h>
#include "cuda_safe.h"

#define GPU_COUNT		1 //  number of GPUs
#define GPU_OFFSET		0 // first GPU to be used

struct foo_test {
	unsigned int device;
	unsigned long mem_size;
	unsigned long nelem;
	unsigned int max_iter;
	pthread_t thread;
};

void foo_func(struct foo_test *info);

int main( int argc, char *argv[] )
{
	unsigned long mem_size = 20;
	unsigned long nelem;
	unsigned int g;
	int i;
	unsigned int max_iter= 10;
	struct foo_test *gpu_info;
	
	if( argc > 1 )
		mem_size= atoi(argv[1]);
	mem_size= powl(2, mem_size);
	nelem= mem_size/sizeof(float);

	cuInit(0);

	g= 0;
	gpu_info= (struct foo_test*)malloc(sizeof(struct foo_test)*GPU_COUNT);
	for( i= GPU_OFFSET; i < (GPU_OFFSET+GPU_COUNT); i++ ) {
		gpu_info[g].device= i;
		gpu_info[g].mem_size= mem_size;
		gpu_info[g].nelem= nelem;
		gpu_info[g].max_iter= max_iter;
		foo_func( &gpu_info[g] );
		g++;
	}
	free( gpu_info );

	return EXIT_SUCCESS;
}

void foo_func(struct foo_test *info)
{
	CUdevice cuDevice;
	CUcontext cuContext;
	float *h_data;
	CUdeviceptr d_data;
	double time_in_us;
	struct timeval t0, t1;
	unsigned int flags;
	unsigned int i, x, j;
	unsigned long mem_size;

	flags= 0;
	CU_SAFE_CALL( cuDeviceGet( &cuDevice, info->device ) );
	CU_SAFE_CALL( cuCtxCreate( &cuContext, flags, cuDevice ) );

	CU_SAFE_CALL( cuMemAllocHost( (void**)&h_data, info->mem_size ) );
	CU_SAFE_CALL( cuMemAlloc( &d_data, info->mem_size ) );
	for( i= 0; i < info->nelem; i++) h_data[i]= 1.0f;

	fprintf( stdout, "# max memory(MB)= %ld\n", info->mem_size/(1<<20) );
	fprintf( stdout, "# size(B)  time(us)\n" );
	fflush(stdout);
	for( x= 0; x <= 2; x++ ) {
	for( j= 64; j <= 1024; j=j+64 ) {
		mem_size= j*powl(1024,x);
		CU_SAFE_CALL(cuMemcpyHtoD( d_data, h_data, mem_size ));
		CU_SAFE_CALL( cuCtxSynchronize() );

		gettimeofday( &t0, 0 );
		for( i= 0; i < info->max_iter; i++ )
			CU_SAFE_CALL( cuMemcpyHtoD( d_data, h_data, mem_size ));
		CU_SAFE_CALL( cuCtxSynchronize() );
		gettimeofday( &t1, 0 );
		time_in_us= (t1.tv_sec-t0.tv_sec)*1e6+(t1.tv_usec-t0.tv_usec);
		fprintf( stdout, "%10ld %10.5f\n", mem_size,
				time_in_us/(info->max_iter) );
		fflush(stdout);
	}}

	CU_SAFE_CALL( cuMemFree( d_data ) );
	CU_SAFE_CALL( cuMemFreeHost( h_data ) );
	CU_SAFE_CALL( cuCtxDetach( cuContext ) );
}

