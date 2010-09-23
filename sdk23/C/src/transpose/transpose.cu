/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */
 
  
/* Matrix transpose with Cuda 
 * Host code.

 * This example transposes arbitrary-size matrices.  It compares a naive
 * transpose kernel that suffers from non-coalesced writes, to an optimized
 * transpose with fully coalesced memory access and no bank conflicts.  On 
 * a G80 GPU, the optimized transpose can be more than 10x faster for large
 * matrices.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil_inline.h>

// includes, kernels
#include <transpose_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);
extern "C" void computeGold( float* reference, float* idata, 
                         const unsigned int size_x, const unsigned int size_y );

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runTest( argc, argv);

    cutilExit(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    // size of the matrix
#ifdef __DEVICE_EMULATION__
    const unsigned int size_x = 32;
    const unsigned int size_y = 128;
#else
    const unsigned int size_x = 256;
    const unsigned int size_y = 4096;
#endif
    // size of memory required to store the matrix
    const unsigned int mem_size = sizeof(float) * size_x * size_y;
    
    unsigned int timer;
    cutCreateTimer(&timer);

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );

    // allocate host memory
    float* h_idata = (float*) malloc(mem_size);
    // initalize the memory
    srand(15235911);
    for( unsigned int i = 0; i < (size_x * size_y); ++i) 
    {
        h_idata[i] = (float) i;    // rand(); 
    }

    // allocate device memory
    float* d_idata;
    float* d_odata;
    cutilSafeCall( cudaMalloc( (void**) &d_idata, mem_size));
    cutilSafeCall( cudaMalloc( (void**) &d_odata, mem_size));

    // copy host memory to device
    cutilSafeCall( cudaMemcpy( d_idata, h_idata, mem_size,
                                cudaMemcpyHostToDevice) );

    // setup execution parameters
    dim3 grid(size_x / BLOCK_DIM, size_y / BLOCK_DIM, 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);

    // warmup so we don't time CUDA startup
    transpose_naive<<< grid, threads >>>(d_odata, d_idata, size_x, size_y);
    transpose<<< grid, threads >>>(d_odata, d_idata, size_x, size_y);

	// synchronize here, so we make sure that we don't count any time from the asynchronize kernel launches.
	cudaThreadSynchronize();

    int numIterations = 1;

    printf("Transposing a %d by %d matrix of floats...\n", size_x, size_y);

    // execute the kernel
    cutStartTimer(timer);
    for (int i = 0; i < numIterations; ++i)
    {
        transpose_naive<<< grid, threads >>>(d_odata, d_idata, size_x, size_y);
    }
    cudaThreadSynchronize();
    cutStopTimer(timer);
    float naiveTime = cutGetTimerValue(timer);

    // execute the kernel
    
    cutResetTimer(timer);
    cutStartTimer(timer);
    for (int i = 0; i < numIterations; ++i)
    {
        transpose<<< grid, threads >>>(d_odata, d_idata, size_x, size_y);
    }
    cudaThreadSynchronize();
    cutStopTimer(timer);
    float optimizedTime = cutGetTimerValue(timer);

    printf("Naive transpose average time:     %0.3f ms\n", naiveTime / numIterations);
    printf("Optimized transpose average time: %0.3f ms\n\n", optimizedTime / numIterations);

    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");

    // copy result from device to    host
    float* h_odata = (float*) malloc(mem_size);
    cutilSafeCall( cudaMemcpy( h_odata, d_odata, mem_size,
                                cudaMemcpyDeviceToHost) );

    // compute reference solution
    float* reference = (float*) malloc( mem_size);

    computeGold( reference, h_idata, size_x, size_y);

    // check result
    CUTBoolean res = cutComparef( reference, h_odata, size_x * size_y);
    printf(    "Test %s\n", (1    == res)    ? "PASSED" : "FAILED");

    // cleanup memory
    free(h_idata);
    free(h_odata);
    free( reference);
    cutilSafeCall(cudaFree(d_idata));
    cutilSafeCall(cudaFree(d_odata));
    cutilCheckError( cutDeleteTimer(timer));

    cudaThreadExit();
}
