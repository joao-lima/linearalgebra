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
 
 /*
 * This sample implements 64-bin histogram calculation
 * of arbitrary-sized 8-bit data array
 */



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>


#include "histogram_common.h"

#ifdef __DEVICE_EMULATION__
const uint numRuns = 1;
#else
const uint numRuns = 10;
#endif


int main(int argc, char **argv){
    uchar *h_Data;
    uint  *h_HistogramCPU, *h_HistogramGPU;
    uchar *d_Data;
    uint  *d_Histogram;

    const uint     byteCount = 128 * 1048576;
    uint hTimer;

    //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );

    cutilCheckError(cutCreateTimer(&hTimer));

    printf("Initializing data...\n");
        printf("...allocating CPU memory.\n");
            h_Data         = (uchar *)malloc(byteCount);
            h_HistogramCPU = (uint  *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
            h_HistogramGPU = (uint  *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));

        printf("...generating input data\n");
            srand(2009);
            for(uint i = 0; i < byteCount; i++) 
                h_Data[i] = rand() % 256;

        printf("...allocating GPU memory and copying input data\n\n");
            cutilSafeCall( cudaMalloc((void **)&d_Data, byteCount  ) );
            cutilSafeCall( cudaMalloc((void **)&d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint)  ) );
            cutilSafeCall( cudaMemcpy(d_Data, h_Data, byteCount, cudaMemcpyHostToDevice) );

    {
        printf("Starting up 64-bin histogram...\n\n");
            initHistogram64();

        printf("Running 64-bin GPU histogram for %u bytes (%u runs)...\n", byteCount , numRuns);
            cutilSafeCall( cudaThreadSynchronize() );
            cutilCheckError( cutResetTimer(hTimer) );
            cutilCheckError( cutStartTimer(hTimer) );
            for(uint iter = 0; iter < numRuns; iter++)
                histogram64(d_Histogram, d_Data, byteCount);
            cutilSafeCall( cudaThreadSynchronize() );
            cutilCheckError(  cutStopTimer(hTimer));
            double timerValue = cutGetTimerValue(hTimer) / (double)numRuns;
        printf("histogram64() time (average) : %f msec //%f MB/sec\n\n", timerValue, ((double)byteCount * 1e-6) / (timerValue * 0.001));

        printf("Validating GPU results...\n");
            printf("...reading back GPU results\n");
                cutilSafeCall( cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM64_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost) );

            printf("...histogram64CPU()\n");
               histogram64CPU(
                    h_HistogramCPU,
                    h_Data,
                    byteCount
                );

            printf("...comparing the results...\n");
                int flag = 1;
                for(uint i = 0; i < HISTOGRAM64_BIN_COUNT; i++)
                    if(h_HistogramGPU[i] != h_HistogramCPU[i]) flag = 0;
            printf(flag ? "TEST PASSED\n\n" : "TEST FAILED\n\n");

        printf("Shutting down 64-bin histogram...\n\n");
            closeHistogram64();
    }


    {
        printf("Starting up 256-bin histogram...\n\n");
            initHistogram256();

        printf("Running 256-bin GPU histogram for %u bytes (%u runs)...\n", byteCount, numRuns);
            cutilSafeCall( cudaThreadSynchronize() );
            cutilCheckError( cutResetTimer(hTimer) );
            cutilCheckError( cutStartTimer(hTimer) );
            for(uint iter = 0; iter < numRuns; iter++)
                histogram256(d_Histogram, d_Data, byteCount);
            cutilSafeCall( cudaThreadSynchronize() );
            cutilCheckError(  cutStopTimer(hTimer));
            double timerValue = cutGetTimerValue(hTimer) / (double)numRuns;
        printf("histogram256() time (average) : %f msec //%f MB/sec\n\n", timerValue, (byteCount * 1e-6) / (timerValue * 0.001));

        printf("Validating GPU results...\n");
            printf("...reading back GPU results\n");
                cutilSafeCall( cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost) );

            printf("...histogram256CPU()\n");
                histogram256CPU(
                    h_HistogramCPU,
                    h_Data,
                    byteCount
                );

            printf("...comparing the results\n");
                int flag = 1;
                for(uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
                    if(h_HistogramGPU[i] != h_HistogramCPU[i]) flag = 0;
        printf(flag ? "TEST PASSED\n\n" : "TEST FAILED\n\n");

        printf("Shutting down 256-bin histogram...\n\n");
            closeHistogram256();
    }



    printf("Shutting down...\n");
        cutilCheckError(cutDeleteTimer(hTimer));
        cutilSafeCall( cudaFree(d_Histogram) );
        cutilSafeCall( cudaFree(d_Data) );
        free(h_HistogramGPU);
        free(h_HistogramCPU);
        free(h_Data);

    cudaThreadExit();
    cutilExit(argc, argv);
}
