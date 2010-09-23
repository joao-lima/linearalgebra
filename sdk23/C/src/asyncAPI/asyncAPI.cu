/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 *
 *
 * This sample illustrates the usage of CUDA events for both GPU timing and
 * overlapping CPU and GPU execution.  Events are insterted into a stream
 * of CUDA calls.  Since CUDA stream calls are asynchronous, the CPU can
 * perform computations while GPU is executing (including DMA memcopies
 * between the host and device).  CPU can query CUDA events to determine
 * whether GPU has completed tasks.
 *
*/

#include <stdio.h>
#include <cutil_inline.h>

__global__ void increment_kernel(int *g_data, int inc_value)
{ 
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   g_data[idx] = g_data[idx] + inc_value;
}

int correct_output(int *data, const int n, const int x)
{
    for(int i = 0; i < n; i++)
        if(data[i] != x)
            return 0;
    return 1;
}

int main(int argc, char *argv[])
{
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
	    cutilDeviceInit(argc, argv);
	else
	    cudaSetDevice( cutGetMaxGflopsDeviceId());

    int n = 16 * 1024 * 1024;
    int nbytes = n * sizeof(int);
    int value = 26;

#ifdef __DEVICE_EMULATION__
    n = 1024;   // reduce workload for emulation
#endif
    
    // allocate host memory
    int *a = 0;
    cutilSafeCall( cudaMallocHost((void**)&a, nbytes) );
    memset(a, 0, nbytes);

    // allocate device memory
    int *d_a=0;
    cutilSafeCall( cudaMalloc((void**)&d_a, nbytes) );
    cutilSafeCall( cudaMemset(d_a, 255, nbytes) );

    // set kernel launch configuration
    dim3 threads = dim3(512, 1);
    dim3 blocks  = dim3(n / threads.x, 1);

    // create cuda event handles
    cudaEvent_t start, stop;
    cutilSafeCall( cudaEventCreate(&start) );
    cutilSafeCall( cudaEventCreate(&stop)  );
    
    unsigned int timer;
    cutilCheckError(  cutCreateTimer(&timer)  );
    cutilCheckError(  cutResetTimer(timer)    );
    cutilSafeCall( cudaThreadSynchronize() );
    float gpu_time = 0.0f;

    // asynchronously issue work to the GPU (all to stream 0)
    cutilCheckError( cutStartTimer(timer) );
        cudaEventRecord(start, 0);
        cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
        increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
        cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
        cudaEventRecord(stop, 0);
    cutilCheckError( cutStopTimer(timer) );

    // have CPU do some work while waiting for stage 1 to finish
    unsigned long int counter=0;
    while( cudaEventQuery(stop) == cudaErrorNotReady )
    {
        counter++;
    }
    cutilSafeCall( cudaEventElapsedTime(&gpu_time, start, stop) );

    // print the cpu and gpu times
    printf("time spent executing by the GPU: %.2f\n", gpu_time);
    printf("time spent by CPU in CUDA calls: %.2f\n", cutGetTimerValue(timer) );
    printf("CPU executed %d iterations while waiting for GPU to finish\n", counter);

    // check the output for correctness
    printf("--------------------------------------------------------------\n");
    if( correct_output(a, n, value) )
        printf("Test PASSED\n");
    else
        printf("Test FAILED\n");

    // release resources
    cutilSafeCall( cudaEventDestroy(start) );
    cutilSafeCall( cudaEventDestroy(stop) );
    cutilSafeCall( cudaFreeHost(a) );
    cutilSafeCall( cudaFree(d_a) );

    cudaThreadExit();

    cutilExit(argc, argv);

    return 0;
}
