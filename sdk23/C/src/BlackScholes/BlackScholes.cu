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
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.
 */



#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <cutil_inline.h>



////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(
    float *h_CallResult,
    float *h_PutResult,
    float *h_StockPrice,
    float *h_OptionStrike,
    float *h_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
);



////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
#include "BlackScholes_kernel.cuh"



////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high){
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}



////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int OPT_N = 4000000;

#ifdef __DEVICE_EMULATION__
const int  NUM_ITERATIONS = 1;
#else
const int  NUM_ITERATIONS = 512;
#endif


const int          OPT_SZ = OPT_N * sizeof(float);
const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
    //'h_' prefix - CPU (host) memory space
    float
        //Results calculated by CPU for reference
        *h_CallResultCPU,
        *h_PutResultCPU,
        //CPU copy of GPU results
        *h_CallResultGPU,
        *h_PutResultGPU,
        //CPU instance of input data
        *h_StockPrice,
        *h_OptionStrike,
        *h_OptionYears;

    //'d_' prefix - GPU (device) memory space
    float
        //Results calculated by GPU
        *d_CallResult,
        *d_PutResult,
        //GPU instance of input data
        *d_StockPrice,
        *d_OptionStrike,
        *d_OptionYears;

    double
        delta, ref, sum_delta, sum_ref, max_delta, L1norm, gpuTime;

    unsigned int hTimer;
    int i;


    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );
		
    cutilCheckError( cutCreateTimer(&hTimer) );

    printf("Initializing data...\n");
        printf("...allocating CPU memory for options.\n");
        h_CallResultCPU = (float *)malloc(OPT_SZ);
        h_PutResultCPU  = (float *)malloc(OPT_SZ);
        h_CallResultGPU = (float *)malloc(OPT_SZ);
        h_PutResultGPU  = (float *)malloc(OPT_SZ);
        h_StockPrice    = (float *)malloc(OPT_SZ);
        h_OptionStrike  = (float *)malloc(OPT_SZ);
        h_OptionYears   = (float *)malloc(OPT_SZ);

        printf("...allocating GPU memory for options.\n");
        cutilSafeCall( cudaMalloc((void **)&d_CallResult,   OPT_SZ) );
        cutilSafeCall( cudaMalloc((void **)&d_PutResult,    OPT_SZ) );
        cutilSafeCall( cudaMalloc((void **)&d_StockPrice,   OPT_SZ) );
        cutilSafeCall( cudaMalloc((void **)&d_OptionStrike, OPT_SZ) );
        cutilSafeCall( cudaMalloc((void **)&d_OptionYears,  OPT_SZ) );

        printf("...generating input data in CPU mem.\n");
        srand(5347);
        //Generate options set
        for(i = 0; i < OPT_N; i++){
            h_CallResultCPU[i] = 0.0f;
            h_PutResultCPU[i]  = -1.0f;
            h_StockPrice[i]    = RandFloat(5.0f, 30.0f);
            h_OptionStrike[i]  = RandFloat(1.0f, 100.0f);
            h_OptionYears[i]   = RandFloat(0.25f, 10.0f);
        }

        printf("...copying input data to GPU mem.\n");
        //Copy options data to GPU memory for further processing
        cutilSafeCall( cudaMemcpy(d_StockPrice,  h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice) );
        cutilSafeCall( cudaMemcpy(d_OptionStrike, h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice) );
        cutilSafeCall( cudaMemcpy(d_OptionYears,  h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice) );
    printf("Data init done.\n");


    printf("Executing Black-Scholes GPU kernel (%i iterations)...\n", NUM_ITERATIONS);
        cutilSafeCall( cudaThreadSynchronize() );
        cutilCheckError( cutResetTimer(hTimer) );
        cutilCheckError( cutStartTimer(hTimer) );
        for(i = 0; i < NUM_ITERATIONS; i++){
            BlackScholesGPU<<<480, 128>>>(
                d_CallResult,
                d_PutResult,
                d_StockPrice,
                d_OptionStrike,
                d_OptionYears,
                RISKFREE,
                VOLATILITY,
                OPT_N
            );
            cutilCheckMsg("BlackScholesGPU() execution failed\n");
        }
        cutilSafeCall( cudaThreadSynchronize() );
        cutilCheckError( cutStopTimer(hTimer) );
        gpuTime = cutGetTimerValue(hTimer) / NUM_ITERATIONS;
    //Both call and put is calculated
    printf("Options count             : %i     \n", 2 * OPT_N);
    printf("BlackScholesGPU() time    : %f msec\n", gpuTime);
    printf("Effective memory bandwidth: %f GB/s\n", ((double)(5 * OPT_N * sizeof(float)) * 1E-9) / (gpuTime * 1E-3));
    printf("Gigaoptions per second    : %f     \n", ((double)(2 * OPT_N) * 1E-9) / (gpuTime * 1E-3));


    printf("Reading back GPU results...\n");
        //Read back GPU results to compare them to CPU results
        cutilSafeCall( cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy(h_PutResultGPU,  d_PutResult,  OPT_SZ, cudaMemcpyDeviceToHost) );


    printf("Checking the results...\n");
        printf("...running CPU calculations.\n");
        //Calculate options values on CPU
        BlackScholesCPU(
            h_CallResultCPU,
            h_PutResultCPU,
            h_StockPrice,
            h_OptionStrike,
            h_OptionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N
        );

        printf("Comparing the results...\n");
        //Calculate max absolute difference and L1 distance
        //between CPU and GPU results
        sum_delta = 0;
        sum_ref   = 0;
        max_delta = 0;
        for(i = 0; i < OPT_N; i++){
            ref   = h_CallResultCPU[i];
            delta = fabs(h_CallResultCPU[i] - h_CallResultGPU[i]);
            if(delta > max_delta) max_delta = delta;
            sum_delta += delta;
            sum_ref   += fabs(ref);
        }
        L1norm = sum_delta / sum_ref;
        printf("L1 norm: %E\n", L1norm);
        printf("Max absolute error: %E\n", max_delta);
    printf((L1norm < 1e-6) ? "TEST PASSED\n" : "TEST FAILED\n");


    printf("Shutting down...\n");
        printf("...releasing GPU memory.\n");
        cutilSafeCall( cudaFree(d_OptionYears)  );
        cutilSafeCall( cudaFree(d_OptionStrike) );
        cutilSafeCall( cudaFree(d_StockPrice)  );
        cutilSafeCall( cudaFree(d_PutResult)    );
        cutilSafeCall( cudaFree(d_CallResult)   );

        printf("...releasing CPU memory.\n");
        free(h_OptionYears);
        free(h_OptionStrike);
        free(h_StockPrice);
        free(h_PutResultGPU);
        free(h_CallResultGPU);
        free(h_PutResultCPU);
        free(h_CallResultCPU);
        cutilCheckError( cutDeleteTimer(hTimer) );
    printf("Shutdown done.\n");

    cudaThreadExit();

    cutilExit(argc, argv);
}
