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

#include <cuda.h>
#include <iostream>
#include "cutil_inline.h"

#include "ptxjit.h"

CUresult initialize(int device, CUcontext *phContext, CUdevice *phDevice, CUmodule *phModule, CUstream *phStream, CUfunction *phKernel)
{
    // Initialize the device and create the context
    cutilDrvSafeCall(cuInit(0));
    cutilDrvSafeCall(cuDeviceGet(phDevice, device));
    cutilDrvSafeCall(cuCtxCreate(phContext, 0, *phDevice));

    // Load the PTX from the string myPtx
    cutilDrvSafeCall(cuModuleLoadDataEx(phModule, myPtx, 0, 0, 0));

    // Locate the kernel entry point
    cutilDrvSafeCall(cuModuleGetFunction(phKernel, *phModule, "_Z8myKernelPi"));

    // Create stream
    cutilDrvSafeCall(cuStreamCreate(phStream, 0));
    
    return CUDA_SUCCESS;
}

int main(int argc, char **argv)
{
    const unsigned int nThreads = 256;
    const unsigned int nBlocks  = 64;
    const size_t memSize = nThreads * nBlocks * sizeof(int);

    CUcontext    hContext = 0;
    CUdevice     hDevice  = 0;
    CUmodule     hModule  = 0;
    CUstream     hStream  = 0;
    CUfunction   hKernel  = 0;
    CUdeviceptr  d_data   = 0;
    int         *h_data   = 0;
    int          iDevice  = 0;

    // Use command-line specified CUDA device, otherwise use device 0
    if (! cutGetCmdLineArgumenti(argc, (const char**)argv, "device", &iDevice))
        iDevice = 0;

    // Initialize the device and get a handle to the kernel
    cutilDrvSafeCall(initialize(iDevice, &hContext, &hDevice, &hModule, &hStream, &hKernel));

    // Allocate memory on host and device
    if ((h_data = (int *)malloc(memSize)) == NULL)
    {
        std::cerr << "Could not allocate host memory" << std::endl;
        exit(-1);
    }
    cutilDrvSafeCall(cuMemAlloc(&d_data, memSize));

    // Set the kernel parameters
    cutilDrvSafeCall(cuFuncSetBlockShape(hKernel, nThreads, 1, 1));
    int paramOffset = 0;
    cutilDrvSafeCall(cuParamSetv(hKernel, paramOffset, &d_data, sizeof(d_data)));
    paramOffset += sizeof(d_data);
    cutilDrvSafeCall(cuParamSetSize(hKernel, paramOffset));

    // Launch the kernel
    cutilDrvSafeCall(cuLaunchGrid(hKernel, nBlocks, 1));
    std::cout << "CUDA kernel launched" << std::endl;
    
    // Copy the result back to the host
    cutilDrvSafeCall(cuMemcpyDtoH(h_data, d_data, memSize));

    // Check the result
    bool dataGood = true;
    for (int i = 0 ; dataGood && i < nBlocks * nThreads ; i++)
    {
        if (h_data[i] != i)
        {
            std::cerr << "Error at " << i << std::endl;
            dataGood = false;
        }
    }
    if (dataGood)
    {
        std::cout << "Test PASSED" << std::endl;
    }
    else
    {
        std::cout << "Test FAILED" << std::endl;
    }

    // Cleanup
    if (d_data)
    {
        cutilDrvSafeCall(cuMemFree(d_data));
        d_data = 0;
    }
    if (h_data)
    {
        free(h_data);
        h_data = 0;
    }
    if (hModule)
    {
        cutilDrvSafeCall(cuModuleUnload(hModule));
        hModule = 0;
    }
    if (hStream)
    {
        cutilDrvSafeCall(cuStreamDestroy(hStream));
        hStream = 0;
    }
    if (hContext)
    {
        cutilDrvSafeCall(cuCtxDestroy(hContext));
        hContext = 0;
    }

    cutilExit(argc,argv);
    return 0;
}
