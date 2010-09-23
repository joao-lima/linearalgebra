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
 
 
/* Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample revisits matrix multiplication with CUDA task. The code of matrix
 * multiplication is exactly the same as in matrixMulDrv sample of this SDK. 
 * This sample, however, demonstrates how to link CUDA driver at runtime and 
 * how to perform JIT (just-in-time) compilation of CUDA kernel from PTX image,
 * stored in memory.
 * 
 * For more details on acquiring auto-generated sources refer README.TXT file 
 * in "extras" directory.
 *
 * Unlike CUBLAS, the sample doesn't address high-performance matrix 
 * multiplication. 
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, CUDA
#include "cuda_drvapi_dynlink.h"
#include "cutil_short.h"

// includes, project
#include "matrixMul.h"
#include "matrixMul_ptxdump.h"


extern "C" void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

#if defined _MSC_VER
    #pragma warning (disable : 4312)
#endif


////////////////////////////////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////////////////////////////
CUcontext g_cuContext;


////////////////////////////////////////////////////////////////////////////////
// Allocates a matrix with random float entries
////////////////////////////////////////////////////////////////////////////////
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = rand() / (float)RAND_MAX;
    }
}


////////////////////////////////////////////////////////////////////////////////
// CUDA driver runtime linking and initialization
////////////////////////////////////////////////////////////////////////////////
CUresult initCUDA(CUfunction *pMatrixMul)
{
    CUresult status;
    CUdevice cuDevice;
    CUmodule cuModule;
    CUfunction cuFunction;

    // link to cuda driver dynamically 
    status = cuInit(0);
    if (CUDA_SUCCESS != status) 
    {
        fprintf(stderr, "Fatal error: Couldn't link to CUDA driver\n");
        exit(-1);
    }

    // get cuda-capable device count
    int deviceCount = 0;
    cutilDrvSafeCallNoSync(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0) 
    {
        fprintf(stderr, "Fatal error: no devices supporting CUDA\n");
        exit(-1);
    }

    // pick up device with zero ordinal
    cutilDrvSafeCallNoSync(cuDeviceGet(&cuDevice, 0));
	
    // create context for picked device
    status = cuCtxCreate(&g_cuContext, 0, cuDevice);
    if (CUDA_SUCCESS != status)
    {
        goto Error;
    }

    // setup JIT compilation options and perform compilation
    if (sizeof(int) == sizeof(void *))
    {
        // in this branch we use compilation with parameters
        const unsigned int jitNumOptions = 2;
        CUjit_option *jitOptions = new CUjit_option[jitNumOptions];
        void **jitOptVals = new void*[jitNumOptions];

        // set up size of compilation log buffer
        jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        int jitLogBufferSize = 1024;
        jitOptVals[0] = (void *)jitLogBufferSize;

        // set up pointer to the compilation log buffer
        jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
        char *jitLogBuffer = new char[jitLogBufferSize];
        jitOptVals[1] = jitLogBuffer;

        // compile with set parameters
        printf("> Compiling CUDA module\n");
        status = cuModuleLoadDataEx(&cuModule, matrixMul_ptxdump, jitNumOptions, jitOptions, (void **)jitOptVals);

        printf("> PTX JIT log:\n%s\n", jitLogBuffer);
        
        delete [] jitOptions;
        delete [] jitOptVals;
        delete [] jitLogBuffer;
    }
    else
    {
        // compile with default parameters
        status = cuModuleLoadData(&cuModule, matrixMul_ptxdump);
    }

    if (CUDA_SUCCESS != status) 
    {
        printf ("Error while compiling PTX\n");
        goto Error;
    }

    // retrieve CUDA function from the compiled module
    status = cuModuleGetFunction(&cuFunction, cuModule, "matrixMul");
    if (CUDA_SUCCESS != status)
    {
        goto Error;
    }

    *pMatrixMul = cuFunction;
    return CUDA_SUCCESS;

Error:
    cuCtxDetach(g_cuContext);
    return status;
}


////////////////////////////////////////////////////////////////////////////////
// Entry point
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    // initialize CUDA
    CUfunction matrixMul = NULL;
    cutilDrvSafeCallNoSync(initCUDA(&matrixMul));

    // set seed for rand()
    srand(2006);

    // allocate host memory for matrices A and B
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);
    unsigned int size_B = WB * HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);

    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

    // allocate device memory
    CUdeviceptr d_A;
    cutilDrvSafeCallSync(cuMemAlloc(&d_A, mem_size_A));
    CUdeviceptr d_B;
    cutilDrvSafeCallSync(cuMemAlloc(&d_B, mem_size_B)); 

    // copy host memory to device
    cutilDrvSafeCallSync(cuMemcpyHtoD(d_A, h_A, mem_size_A));
    cutilDrvSafeCallSync(cuMemcpyHtoD(d_B, h_B, mem_size_B));

    // allocate device memory for result
    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    CUdeviceptr d_C;
    cutilDrvSafeCallSync(cuMemAlloc(&d_C, mem_size_C));

    // allocate mem for the result on host side
    float* h_C = (float*) malloc(mem_size_C);

    // setup execution parameters
    int offset = 0;
    cutilDrvSafeCallNoSync(cuParamSetv(matrixMul, offset, &d_C, sizeof(CUdeviceptr)));
    offset += sizeof(CUdeviceptr);

    cutilDrvSafeCallNoSync(cuParamSetv(matrixMul, offset, &d_A, sizeof(CUdeviceptr)));
    offset += sizeof(CUdeviceptr);

    cutilDrvSafeCallNoSync(cuParamSetv(matrixMul, offset, &d_B, sizeof(CUdeviceptr)));
    offset += sizeof(CUdeviceptr);

    int Matrix_Width_A = WA;
    int Matrix_Width_B = WB;

    cutilDrvSafeCallNoSync(cuParamSeti(matrixMul, offset, Matrix_Width_A)); 
    offset += sizeof(Matrix_Width_A);

    cutilDrvSafeCallNoSync(cuParamSeti(matrixMul, offset, Matrix_Width_B)); 
    offset += sizeof(Matrix_Width_B);

    cutilDrvSafeCallNoSync(cuParamSetSize(matrixMul, offset));
    cutilDrvSafeCallNoSync(cuFuncSetBlockShape(matrixMul, BLOCK_SIZE, BLOCK_SIZE, 1));
    cutilDrvSafeCallNoSync(cuFuncSetSharedSize(matrixMul, 2*BLOCK_SIZE*BLOCK_SIZE*sizeof(float)));

    // set execution configuration for the CUDA kernel
    cutilDrvSafeCallSync(cuLaunchGrid(matrixMul, WC / BLOCK_SIZE, HC / BLOCK_SIZE));

    // copy result from device to host
    cutilDrvSafeCallSync(cuMemcpyDtoH((void *) h_C, d_C, mem_size_C));

    // compute reference solution
    float* reference = (float*) malloc(mem_size_C);
    computeGold(reference, h_A, h_B, HA, WA, WB);

    // check result
    float diff=0.0f;
    for (unsigned int i=0; i<size_C; i++)
    {
        float tmp = reference[i] - h_C[i];
        diff += tmp*tmp;
    }

    int res = (diff / (float)size_C < 1e-6f);
    printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(reference);
    cutilDrvSafeCallSync(cuMemFree(d_A));
    cutilDrvSafeCallSync(cuMemFree(d_B));
    cutilDrvSafeCallSync(cuMemFree(d_C));
    cutilDrvSafeCallNoSync(cuCtxDetach(g_cuContext));

    cutilExit(argc, argv);
}
