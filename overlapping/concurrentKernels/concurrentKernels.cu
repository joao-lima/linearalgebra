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

const char *sSDKsample = "concurrentKernels";

#define CUDA_SAFE_CALL(call) do {                                 \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }while(0)


#define	NTASKS	4
#define	NSTREAMS	4

__global__ void add1( float* array, unsigned int size )
{
  const unsigned int per_thread = size / blockDim.x;
  unsigned int i = threadIdx.x * per_thread;

  unsigned int j = size;
  if (threadIdx.x != (blockDim.x - 1)) j = i + per_thread;

  for (; i < j; ++i) ++array[i];
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
    unsigned int nstreams = NSTREAMS;
    unsigned int ntasks = NTASKS;
    unsigned int nevents = ntasks * 2;
    float *h_data[NTASKS], *d_data[NTASKS];
    float elapsed_time= 0;

    printf("[ %s ]\n", sSDKsample);
    cuda_device = 0;
    cudaDeviceProp deviceProp;
    CUDA_SAFE_CALL( cudaGetDevice(&cuda_device));	

    CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, cuda_device) );
    if( (deviceProp.concurrentKernels == 0 ))
        printf("> GPU does not support concurrent kernel execution, kernel runs will be serialized\n");

    printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n", deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount); 

    cudaStream_t *streams = (cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));
    cudaEvent_t *events = (cudaEvent_t*) malloc(nevents * sizeof(cudaEvent_t));
    for(int i = 0; i < nstreams; i++)
	CUDA_SAFE_CALL( cudaStreamCreate(&(streams[i])) );
    for(int i = 0; i < nevents; i++)
        CUDA_SAFE_CALL( cudaEventCreateWithFlags(&(events[i]), cudaEventDisableTiming) );

    for( int i= 0; i < ntasks; i++ ) {
	CUDA_SAFE_CALL( cudaMallocHost((void**)&h_data[i], mem_size) ); 
	CUDA_SAFE_CALL( cudaMalloc((void**)&d_data[i], mem_size) );
	for( int j= 0; j < (mem_size/sizeof(float)); j++ )
		h_data[i][j]= 1.0f;
    }
    // create CUDA event handles
    cudaEvent_t start_event, stop_event;
    CUDA_SAFE_CALL( cudaEventCreate(&start_event) );
    CUDA_SAFE_CALL( cudaEventCreate(&stop_event) );
	
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    cudaEventRecord(start_event, 0);
    // queue nkernels in separate streams and record when they are done
    for( int i=0; i < ntasks; ++i) {
	CUDA_SAFE_CALL( cudaMemcpyAsync( d_data[i], h_data[i], mem_size,
			cudaMemcpyHostToDevice, streams[0] ));
	CUDA_SAFE_CALL( cudaEventRecord( events[i*2], streams[0] ) );

	CUDA_SAFE_CALL( cudaStreamWaitEvent( streams[1], events[i*2], 0) );
        add1<<<1,256,0,streams[1]>>>(d_data[i], (mem_size/sizeof(float)) );
	CUDA_SAFE_CALL( cudaEventRecord( events[i*2+1], streams[1] ) );

	CUDA_SAFE_CALL( cudaStreamWaitEvent( streams[2], events[i*2+1], 0) );
	CUDA_SAFE_CALL( cudaMemcpyAsync( h_data[i], d_data[i], mem_size,
			cudaMemcpyDeviceToHost, streams[2] ) );
    }

    // in this sample we just wait until the GPU is done
    CUDA_SAFE_CALL( cudaEventRecord(stop_event, 0) );
    CUDA_SAFE_CALL( cudaEventSynchronize(stop_event) );
    CUDA_SAFE_CALL( cudaEventElapsedTime(&elapsed_time,
		    start_event, stop_event) );
    
    printf("Measured time for sample = %.3fs\n", elapsed_time/1000.0f);

    for( int i= 0; i < ntasks; i++ )
	    if( check( h_data[i], mem_size/sizeof(float), 2.0) )
		    fprintf(stdout, "ERROR at task %d\n", i ); fflush(stdout);
    
    // release resources
    for(int i = 0; i < nstreams; i++)
		cudaStreamDestroy(streams[i]);

    for(int i = 0; i < nevents; i++)
		cudaEventDestroy(events[i]);
    for( int i= 0; i < ntasks; i++ ) {
	    cudaFreeHost(h_data[i]);
	    cudaFree(d_data[i]);
    }

    free(streams);
    free(events);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    cudaThreadExit();
}
