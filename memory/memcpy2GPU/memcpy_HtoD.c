
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>

#include <cuda.h>
#include "cuda_safe.h"

#define GPU_COUNT		2 //  number of GPUs
#define GPU_OFFSET		0 // first GPU to be used

struct foo_test {
	unsigned int device;
	unsigned int mem_size;
	unsigned int nelem;
	unsigned int max_iter;
	pthread_t thread;
};

void *foo_thread_func(void *v);
int check( const float *data, const float v, const unsigned int n );

static pthread_barrier_t g_barrier;

int main( int argc, char *argv[] )
{
	unsigned int mem_size = (1 << 24);
	unsigned int nelem;
	unsigned int g;
	int i;
	unsigned int max_iter= 10;
	struct foo_test *gpu_info;
	
	if( argc > 1 )
		mem_size= (1 << atoi(argv[1]));
	nelem= mem_size/sizeof(float);

	cuInit(0);

	if( pthread_barrier_init( &g_barrier, NULL, GPU_COUNT ) != 0 ) {
        	fprintf( stdout, "Barrier error in file '%s' in line %i.\n",
			__FILE__, __LINE__ );
		exit( EXIT_FAILURE );
	}
	g= 0;
	gpu_info= (struct foo_test*)malloc(sizeof(struct foo_test)*GPU_COUNT);
	for( i= GPU_OFFSET; i < (GPU_OFFSET+GPU_COUNT); i++ ) {
	//for( i= (GPU_OFFSET+GPU_COUNT-1); i >= GPU_OFFSET; i-- ) {
		gpu_info[g].device= i;
		gpu_info[g].mem_size= mem_size;
		gpu_info[g].nelem= nelem;
		gpu_info[g].max_iter= max_iter;
		if(pthread_create(&gpu_info[g].thread, NULL, foo_thread_func,
					(void*)(gpu_info+g))){
			fprintf( stdout, "ERROR thread %d\n", i);
			fflush(stdout);
		}
		g++;
	}
	for( i= 0; i < GPU_COUNT; i++ ) {
		pthread_join(gpu_info[i].thread, NULL);
	}

	return EXIT_SUCCESS;
}

void *foo_thread_func(void *v)
{
	struct foo_test *info = (struct foo_test*)v;
	CUdevice cuDevice;
	CUcontext cuContext;
	float *h_data;
	CUdeviceptr d_data;
	CUevent e1, e2;
	float elapsed_time_in_Ms= 0.0f;
	float bandwidth_in_MBs= 0.0f;
	unsigned int flags;
	unsigned int i;

#if 0
	fprintf( stdout, "foo_thread_func device=%d\n", info->device );
	fflush(stdout);
#endif
	flags= 0;
	CU_SAFE_CALL( cuDeviceGet( &cuDevice, info->device ) );
	CU_SAFE_CALL( cuCtxCreate( &cuContext, flags, cuDevice ) );
	CU_SAFE_CALL( cuEventCreate( &e1, 0 ) );
	CU_SAFE_CALL( cuEventCreate( &e2, 0 ) );

	h_data= (float*) malloc( info->mem_size );
	CU_SAFE_CALL( cuMemAlloc( &d_data, info->mem_size ) );
	for( i= 0; i < info->nelem; i++) h_data[i]= 1.0f;

	/* Here, it just try to transfer data at the same time */
	pthread_barrier_wait( &g_barrier );

	CU_SAFE_CALL( cuEventRecord( e1, 0 ) );
	for( i= 0; i < info->max_iter; i++ )
		CU_SAFE_CALL( cuMemcpyHtoD( d_data, h_data, info->mem_size ) );
	//CU_SAFE_CALL( cuCtxSynchronize() );
	CU_SAFE_CALL( cuEventRecord( e2, 0 ) );
	CU_SAFE_CALL( cuEventSynchronize( e2 ) );

	CU_SAFE_CALL( cuEventElapsedTime( &elapsed_time_in_Ms, e1, e2 ) );
	bandwidth_in_MBs= (1e3f * info->max_iter * info->mem_size)  / 
	       	(elapsed_time_in_Ms * (float)(1 << 20));
	fprintf( stdout, "naive gpu= %d size(KB)= %9u time(ms)= %.3f bandwidth(MB/s)= %.1f\n",
		info->device, info->mem_size/(1<<10),
		elapsed_time_in_Ms/(info->max_iter), bandwidth_in_MBs );
	fflush(stdout);

	if( check( h_data, 1.0f, info->nelem) == 0 ){
		fprintf( stdout, "thread%d test FAILED\n", info->device );
		fflush( stdout );
	}

	CU_SAFE_CALL( cuEventDestroy( e1 ) );
	CU_SAFE_CALL( cuEventDestroy( e2 ) );
	CU_SAFE_CALL( cuMemFree( d_data ) );
	CU_SAFE_CALL( cuCtxDetach( cuContext ) );
	pthread_exit(NULL);
}

int check( const float *data, const float v, const unsigned int n )
{
	unsigned int i;

	for( i= 0; i < n; i++ ){
		if( data[i] != v ) {
			fprintf( stdout, "%d %f\n", i, data[i] );
			fflush( stdout );
			return 0;
		}
	}

	return 1;
}
