#include <stdio.h>

#include <cuda.h>

const char *sSDKsample = "concurrentKernels";

#define CUDA_SAFE_CALL(call) do {                                 \
    CUresult err = call;                                                    \
    if( CUDA_SUCCESS != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %d.\n",        \
                __FILE__, __LINE__, err );              \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }while(0)


#define	NTASKS	2

__global__ void add1( float* array, unsigned int size )
{
  const unsigned int per_thread = size / blockDim.x;
  unsigned int i = threadIdx.x * per_thread;

  unsigned int j = size;
  if (threadIdx.x != (blockDim.x - 1)) j = i + per_thread;

  unsigned int k;
  for (; i < j; ++i)
  for(k = 0; k < 10;k++)
	  ++array[i];
}

int check( const float *data, const unsigned int n, const float v )
{
	for( int i= 0; i < n; i++ )
		if( data[i] != v )
			return 1;

	return 0;
}

int main(int argc, char **argv)
{
    int cuda_device = 0;
    unsigned int mem_size = (1 << 28);
    unsigned int ntasks = NTASKS;
    float *h_data[NTASKS];
    CUdeviceptr d_data[NTASKS];
    float elapsed_time= 0;
    unsigned int nmax= 100;

    cuInit(0);

    CUdevice cuDevice;
    CUcontext cuContext;
    unsigned int flags = CU_CTX_SCHED_YIELD;
    cuda_device = 0;
    CUDA_SAFE_CALL( cuDeviceGet(&cuDevice, cuda_device) );	
    CUDA_SAFE_CALL( cuCtxCreate( &cuContext, flags, cuDevice ) );

    for( int i= 0; i < ntasks; i++ ) {
	CUDA_SAFE_CALL( cuMemHostAlloc( (void**)&h_data[i], mem_size,
			CU_MEMHOSTALLOC_PORTABLE) ); 
	CUDA_SAFE_CALL( cuMemAlloc( &d_data[i], mem_size) );
	//CUDA_SAFE_CALL( cuMemsetD16( d_data[i], 0, mem_size ) );
	for( int j= 0; j < (mem_size/sizeof(float)); j++ )
		h_data[i][j]= 1.0f;
    }
    // create CUDA event handles
    CUevent  start_event, stop_event;
    CUDA_SAFE_CALL( cuEventCreate(&start_event, CU_EVENT_DEFAULT) );
    CUDA_SAFE_CALL( cuEventCreate(&stop_event, CU_EVENT_DEFAULT) );
    CUstream s1, s2;
    CUDA_SAFE_CALL( cuStreamCreate(&s1, 0) );
    CUDA_SAFE_CALL( cuStreamCreate(&s2, 0) );

    unsigned int i= 0;
    CUDA_SAFE_CALL( cuCtxSynchronize() );
    cuEventRecord(start_event, 0);
    for(i= 0; i < nmax; i++) {
	    CUDA_SAFE_CALL( cuMemcpyHtoDAsync( d_data[0], h_data[0], mem_size,
			s1 ));
	    CUDA_SAFE_CALL( cuMemcpyDtoHAsync( h_data[1], d_data[1], mem_size,
			   s2) );
	    //cuCtxSynchronize();
    }
    // in this sample we just wait until the GPU is done
    CUDA_SAFE_CALL( cuEventRecord(stop_event, 0) );
    CUDA_SAFE_CALL( cuEventSynchronize(stop_event) );
    CUDA_SAFE_CALL( cuEventElapsedTime(&elapsed_time, start_event, stop_event) );
    printf("Time duplex = %.4f ms\n", elapsed_time/nmax);

    CUDA_SAFE_CALL( cuCtxSynchronize() );
    cuEventRecord(start_event, 0);
    for(i= 0; i < nmax; i++) {
	    CUDA_SAFE_CALL( cuMemcpyHtoDAsync( d_data[0], h_data[0], mem_size,
			s1 ));
	    //cuCtxSynchronize();
    }
    // in this sample we just wait until the GPU is done
    CUDA_SAFE_CALL( cuEventRecord(stop_event, 0) );
    CUDA_SAFE_CALL( cuEventSynchronize(stop_event) );
    CUDA_SAFE_CALL( cuEventElapsedTime(&elapsed_time, start_event, stop_event) );
    printf("Time simple = %.4f ms\n", elapsed_time/nmax);

    for( int i= 0; i < ntasks; i++ ) {
	    cuMemFreeHost(h_data[i]);
	    cuMemFree(d_data[i]);
    }

    cuEventDestroy(start_event);
    cuEventDestroy(stop_event);
    cuStreamDestroy(s1);
    cuStreamDestroy(s2);
    cuCtxPopCurrent(&cuContext);
    return 0;
}
