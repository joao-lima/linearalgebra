
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>

#include "cuda_safe.h"
#include "device.h"

struct device {
	int device;
	CUdevice cuDevice;
	CUcontext cuContext;
};

struct matrices {
	unsigned int N;
	int *h_A, *h_B, *h_C;
       	CUdeviceptr d_A, d_B, d_C;
};

void computeGold( int *C, const int* A, const int* B, unsigned int N )
{
	unsigned int i, j, k;
	int sum;

	for( i = 0; i < N; ++i ){
		for( j = 0; j < N; ++j ){
		    sum = 0;
		    for( k = 0; k < N; ++k ){
			sum += A[i * N + k] * B[k * N + j];
		    }
		    C[i * N + j] = sum;
		}
	}
}

void cuda_init( struct device *dev )
{
/* CUDA context flags:
	CU_CTX_SCHED_AUTO: CUDA will yield threads waiting for results,
		or spin them otherwise
	CU_CTX_SCHED_SPIN: busy wait of context
	CU_CTX_SCHED_YIELD: yield the thread waiting for results.
	CU_CTX_BLOCKING_SYNC: blocks CPU thread at sync
	CU_CTX_MAP_HOST: allows mapped pinned allocations.
*/
	unsigned int flags= 0;
	CU_SAFE_CALL( cuInit( dev->device ) );
	CU_SAFE_CALL( cuDeviceGet( &dev->cuDevice, dev->device ) );
	CU_SAFE_CALL( cuCtxCreate( &dev->cuContext, flags, dev->cuDevice ) );
	//CU_SAFE_CALL( cuCtxPushCurrent( dev->cuContext ) );
}

void cuda_finalize( struct device *dev )
{
	//CU_SAFE_CALL( cuCtxDetach( dev->cuContext ) );
	CU_SAFE_CALL( cuCtxPopCurrent(NULL) );
}

void mm_alloc( struct matrices *mm )
{
	unsigned int nsize= mm->N * mm->N;
	unsigned int flags = 0;
	/* CUDA hostalloc flags
		CU_MEMHOSTALLOC_PORTABLE - all contexts
		CU_MEMHOSTALLOC_DEVICEMAP - mapped memory
		CU_MEMHOSTALLOC_WRITECOMBINED - write-combined
	   */
	CU_SAFE_CALL( cuMemHostAlloc( (void**)(&mm->h_A), nsize*sizeof(int),
		flags ) );
	CU_SAFE_CALL( cuMemHostAlloc( (void**)(&mm->h_B), nsize*sizeof(int),
		flags ) );
	CU_SAFE_CALL( cuMemHostAlloc( (void**)(&mm->h_C), nsize*sizeof(int),
		flags ) );
	CU_SAFE_CALL( cuMemAlloc( &mm->d_A, nsize * sizeof(int) ) );
	CU_SAFE_CALL( cuMemAlloc( &mm->d_B, nsize * sizeof(int) ) );
	CU_SAFE_CALL( cuMemAlloc( &mm->d_C, nsize * sizeof(int) ) );
}

void mm_init( struct matrices *mm )
{
	unsigned int nsize= mm->N * mm->N;
	unsigned int i;

	srand( time(NULL) + getpid() );
	for( i= 0; i < nsize; i++ ){
		mm->h_A[i]= 1 + (int)(10.0 * (rand() / (RAND_MAX + 1.0)));
		mm->h_B[i]= 1 + (int)(10.0 * (rand() / (RAND_MAX + 1.0)));
	}
}

void mm_free( struct matrices *mm )
{
	CU_SAFE_CALL( cuMemFreeHost( mm->h_A ) );
	CU_SAFE_CALL( cuMemFreeHost( mm->h_B ) );
	CU_SAFE_CALL( cuMemFreeHost( mm->h_C ) );
	CU_SAFE_CALL( cuMemFree( mm->d_A ) );
	CU_SAFE_CALL( cuMemFree( mm->d_B ) );
	CU_SAFE_CALL( cuMemFree( mm->d_C ) );
}

void mm_params( struct matrices *mm, CUfunction *cuFunction )
{
	int offset= 0;
	void* ptr;

	ptr= (void*)(size_t)mm->d_C;
	offset= (offset + __alignof(ptr) - 1) & ~(__alignof(ptr) - 1);
	CU_SAFE_CALL( cuParamSetv( *cuFunction, offset, &ptr, sizeof(ptr) ) );
	offset += sizeof(ptr);

	ptr = (void*)(size_t)mm->d_A;
	offset= (offset + __alignof(ptr) - 1) & ~(__alignof(ptr) - 1);
	CU_SAFE_CALL( cuParamSetv( *cuFunction, offset, &ptr, sizeof(ptr) ) );
	offset += sizeof(ptr);

	ptr = (void*)(size_t)mm->d_B;
	offset= (offset + __alignof(ptr) - 1) & ~(__alignof(ptr) - 1);
	CU_SAFE_CALL( cuParamSetv( *cuFunction, offset, &ptr, sizeof(ptr) ) );
	offset += sizeof(ptr);

	int N = mm->N;
	offset= (offset + __alignof(N) - 1) & ~(__alignof(N) - 1);
	CU_SAFE_CALL( cuParamSeti( *cuFunction, offset, N ) );
	offset += sizeof(N);

	CU_SAFE_CALL( cuParamSetSize( *cuFunction, offset) );
}

void mm_launch( struct matrices *mm )
{
	CUfunction cuFunction;
	CUmodule cuModule;
	char module_path[] = "matrixMul_kernel.ptx";

	CU_SAFE_CALL( cuModuleLoad( &cuModule, module_path ) );
	CU_SAFE_CALL( cuModuleGetFunction( &cuFunction, cuModule, "matrixMul" ));
	mm_params( mm, &cuFunction );
	CU_SAFE_CALL( cuFuncSetBlockShape( cuFunction, BLOCK_SIZE, BLOCK_SIZE, 1 ) );
	//CU_SAFE_CALL( cuFuncSetSharedSize( cuFunction,
	//			2*BLOCK_SIZE*BLOCK_SIZE*sizeof(int) ) );
	CU_SAFE_CALL( cuLaunchGrid( cuFunction, mm->N/BLOCK_SIZE,
			       mm->N/BLOCK_SIZE ) );
	//CU_SAFE_CALL( cuCtxSynchronize() );
}

void mm_cpy_HtoD( struct matrices *mm )
{
	unsigned int nsize= mm->N * mm->N;
	CU_SAFE_CALL( cuMemcpyHtoD( mm->d_A, mm->h_A, nsize * sizeof(int) ) );
	CU_SAFE_CALL( cuMemcpyHtoD( mm->d_B, mm->h_B, nsize * sizeof(int) ) );
	//CU_SAFE_CALL( cuMemsetD8( mm->d_C, 0, nsize ) );
}

void mm_cpy_DtoH( struct matrices *mm )
{
	CU_SAFE_CALL( cuMemcpyDtoH( (void*) mm->h_C, mm->d_C, 
			       mm->N * mm->N * sizeof(int) ) );
}

int mm_compare( const int *c1, const int *c2, const unsigned int nsize )
{
	unsigned int i;

	for( i= 0; i < nsize; i++ )
		if( c1[i] != c2[i] ){
			fprintf( stdout, "d[%d] - %d != %d\n", i,
				c1[i], c2[i] );
			return 0;
		}

	return 1;
}

void mm_test( struct matrices *mm )
{
	int *reference;
	unsigned int nsize= mm->N * mm->N;

	reference= (int*)malloc( nsize * sizeof(int) );
	computeGold( reference, mm->h_A, mm->h_B, mm->N );
	if( mm_compare( reference, mm->h_C, nsize ) == 1 )
		fprintf( stdout, "Test PASSED\n" );
	else
		fprintf( stdout, "Test FAILED\n" );
}

int main( int argc, char **argv )
{
	struct device dev;
	struct matrices mm;
	CUevent e1, e2;
	float elapsed_time_in_Ms= 0;
	float bandwidth_in_MBs= 0;
	int i, max_iter= 10;

	if( argc > 1 )
		mm.N = atoi( argv[1] );
	else
		mm.N = 1024;

	dev.device= 0;
	cuda_init( &dev );
	cuEventCreate( &e1, 0 );
	cuEventCreate( &e2, 0 );

	mm_alloc( &mm );
	mm_init( &mm );
	cuEventRecord( e1, 0 );
	for( i= 0; i < max_iter; i++ ){
		mm_cpy_HtoD( &mm );
		mm_launch( &mm );
		mm_cpy_DtoH( &mm );
	}
	cuEventRecord( e2, 0 );
	if( argc > 2 )
		mm_test( &mm );
	mm_free( &mm );

	cuEventElapsedTime( &elapsed_time_in_Ms, e1, e2 );
	cuda_finalize( &dev );
	bandwidth_in_MBs= 2.0f * (1e3f*mm.N * mm.N * sizeof(int) * max_iter) / 
	       	(elapsed_time_in_Ms * (float)(1 << 20));
	fprintf( stdout, "size= %d time(s)= %.3f bandwidth(MB/s)= %.1f\n",
		mm.N, elapsed_time_in_Ms/(1e3f*max_iter), bandwidth_in_MBs );

	return EXIT_SUCCESS;
}
