/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

//
// This sample demonstrates the use of streams for concurrent execution. It also illustrates how to 
// introduce dependencies between CUDA streams with the new cudaStreamWaitEvent function introduced 
// in CUDA 3.2.
//
// Devices of compute capability 1.x will run the kernels one after another
// Devices of compute capability 2.0 or higher can overlap the kernels
//

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


#define	NTASKS	16

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
    unsigned int mem_size = (1 << 27);
    unsigned int ntasks = NTASKS;
    float *h_data[NTASKS];
    CUdeviceptr d_data[NTASKS];
    float elapsed_time= 0;

    cuInit(0);

    CUdevice cuDevice;
    CUcontext cuContext;
    unsigned int flags = CU_CTX_SCHED_YIELD;
    cuda_device = 0;
    CUDA_SAFE_CALL( cuDeviceGet(&cuDevice, cuda_device) );	
    CUDA_SAFE_CALL( cuCtxCreate( &cuContext, flags, cuDevice ) );

    CUstream *streams = (CUstream*) malloc(ntasks * sizeof(CUstream));
    for(int i = 0; i < ntasks; i++)
	CUDA_SAFE_CALL( cuStreamCreate(&(streams[i]),0 ) );

    for( int i= 0; i < ntasks; i++ ) {
	CUDA_SAFE_CALL( cuMemHostAlloc( (void**)&h_data[i], mem_size,
			CU_MEMHOSTALLOC_PORTABLE) ); 
	CUDA_SAFE_CALL( cuMemAlloc( &d_data[i], mem_size) );
	for( int j= 0; j < (mem_size/sizeof(float)); j++ )
		h_data[i][j]= 1.0f;
    }
    // create CUDA event handles
    CUevent  start_event, stop_event;
    CUDA_SAFE_CALL( cuEventCreate(&start_event, CU_EVENT_DEFAULT) );
    CUDA_SAFE_CALL( cuEventCreate(&stop_event, CU_EVENT_DEFAULT) );
    CUevent e_k;
    CUevent e_HtoD;
    CUDA_SAFE_CALL( cuEventCreate(&e_k, CU_EVENT_DISABLE_TIMING) );
    CUDA_SAFE_CALL( cuEventCreate(&e_HtoD, CU_EVENT_DISABLE_TIMING) );

    unsigned int i_HtoD= 0, i_DtoH= 0, i_k= 0, i= 0;
    CUDA_SAFE_CALL( cuCtxSynchronize() );
    cuEventRecord(start_event, 0);
    CUDA_SAFE_CALL( cuMemcpyHtoDAsync( d_data[i], h_data[i], mem_size,
		streams[0] ));
    CUDA_SAFE_CALL( cuEventRecord(e_HtoD, streams[0]) );
    i_HtoD++;
    while( i < ntasks ) {
	if( i_k < ntasks ) {
		if( cuEventQuery(e_HtoD) == CUDA_SUCCESS ) {
			fprintf(stdout,"kernel (%d)\n", i_k);fflush(stdout);
			add1<<<1,256,0,streams[1]>>>( (float*)d_data[i_k],
					(mem_size/sizeof(float)) );
			CUDA_SAFE_CALL( cuEventRecord(e_k, streams[1]) );
			i_k++;
		}

	}

	if( i_HtoD < ntasks ) {
		if( cuEventQuery(e_HtoD) == CUDA_SUCCESS ) {
		fprintf(stdout,"HtoD (%d)\n", i_HtoD);fflush(stdout);
		CUDA_SAFE_CALL( cuMemcpyHtoDAsync( d_data[i_HtoD],
		       	h_data[i_HtoD], mem_size, streams[0] ) );
		CUDA_SAFE_CALL( cuEventRecord(e_HtoD, streams[0]) );
		i_HtoD++;
		}
	}

	if( i_DtoH < ntasks ){
		if( i_k > 0 &&  cuEventQuery(e_k) == CUDA_SUCCESS ) {
			fprintf(stdout,"DtoH (%d)\n", i_DtoH);fflush(stdout);
			CUDA_SAFE_CALL( cuMemcpyDtoHAsync( h_data[i_DtoH],
				d_data[i_DtoH], mem_size, streams[2]));
			i_DtoH++;
			i++;
		}
	}
    }

    // in this sample we just wait until the GPU is done
    CUDA_SAFE_CALL( cuEventRecord(stop_event, 0) );
    CUDA_SAFE_CALL( cuEventSynchronize(stop_event) );
    CUDA_SAFE_CALL( cuEventElapsedTime(&elapsed_time, start_event, stop_event) );
    
    printf("Measured time for sample = %.3fs\n", elapsed_time/1000.0f);

    for( int i= 0; i < ntasks; i++ )
	    if( check( h_data[i], mem_size/sizeof(float), 11) )
		    fprintf(stdout, "ERROR at task %d\n", i ); fflush(stdout);
    
    // release resources
    for(int i = 0; i < ntasks; i++)
		cuStreamDestroy(streams[i]);

    for( int i= 0; i < ntasks; i++ ) {
	    cuMemFreeHost(h_data[i]);
	    cuMemFree(d_data[i]);
    }

    free(streams);
    cuEventDestroy(start_event);
    cuEventDestroy(stop_event);
    cuEventDestroy(e_k);
    cuEventDestroy(e_HtoD);
    cuCtxPopCurrent(&cuContext);
    return 0;
}
