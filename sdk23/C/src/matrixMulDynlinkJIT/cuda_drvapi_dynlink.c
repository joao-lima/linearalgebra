/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  Users and possessors of this source code 
 * are hereby granted a nonexclusive, royalty-free license to use this code 
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as 
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer  software"  and "commercial computer software 
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein. 
 *
 * Any use of this source code in individual and commercial software must 
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#include <stdio.h>
#include "cuda_drvapi_dynlink.h"

tcuInit                         *_cuInit;
tcuDriverGetVersion             *cuDriverGetVersion;
tcuDeviceGet                    *cuDeviceGet;
tcuDeviceGetCount               *cuDeviceGetCount;
tcuDeviceGetName                *cuDeviceGetName;
tcuDeviceComputeCapability      *cuDeviceComputeCapability;
tcuDeviceTotalMem               *cuDeviceTotalMem;
tcuDeviceGetProperties          *cuDeviceGetProperties;
tcuDeviceGetAttribute           *cuDeviceGetAttribute;
tcuCtxCreate                    *cuCtxCreate;
tcuCtxDestroy                   *cuCtxDestroy;
tcuCtxAttach                    *cuCtxAttach;
tcuCtxDetach                    *cuCtxDetach;
tcuCtxPushCurrent               *cuCtxPushCurrent;
tcuCtxPopCurrent                *cuCtxPopCurrent;
tcuCtxGetDevice                 *cuCtxGetDevice;
tcuCtxSynchronize               *cuCtxSynchronize;
tcuModuleLoad                   *cuModuleLoad;
tcuModuleLoadData               *cuModuleLoadData;
tcuModuleLoadDataEx             *cuModuleLoadDataEx;
tcuModuleLoadFatBinary          *cuModuleLoadFatBinary;
tcuModuleUnload                 *cuModuleUnload;
tcuModuleGetFunction            *cuModuleGetFunction;
tcuModuleGetGlobal              *cuModuleGetGlobal;
tcuModuleGetTexRef              *cuModuleGetTexRef;
tcuMemGetInfo                   *cuMemGetInfo;
tcuMemAlloc                     *cuMemAlloc;
tcuMemAllocPitch                *cuMemAllocPitch;
tcuMemFree                      *cuMemFree;
tcuMemGetAddressRange           *cuMemGetAddressRange;
tcuMemAllocHost                 *cuMemAllocHost;
tcuMemFreeHost                  *cuMemFreeHost;
tcuMemHostAlloc                 *cuMemHostAlloc;
tcuMemHostGetDevicePointer      *cuMemHostGetDevicePointer;
tcuMemHostGetFlags              *cuMemHostGetFlags;
tcuMemcpyHtoD                   *cuMemcpyHtoD;
tcuMemcpyDtoH                   *cuMemcpyDtoH;
tcuMemcpyDtoD                   *cuMemcpyDtoD;
tcuMemcpyDtoA                   *cuMemcpyDtoA;
tcuMemcpyAtoD                   *cuMemcpyAtoD;
tcuMemcpyHtoA                   *cuMemcpyHtoA;
tcuMemcpyAtoH                   *cuMemcpyAtoH;
tcuMemcpyAtoA                   *cuMemcpyAtoA;
tcuMemcpy2D                     *cuMemcpy2D;
tcuMemcpy2DUnaligned            *cuMemcpy2DUnaligned;
tcuMemcpy3D                     *cuMemcpy3D;
tcuMemcpyHtoDAsync              *cuMemcpyHtoDAsync;
tcuMemcpyDtoHAsync              *cuMemcpyDtoHAsync;
tcuMemcpyHtoAAsync              *cuMemcpyHtoAAsync;
tcuMemcpyAtoHAsync              *cuMemcpyAtoHAsync;
tcuMemcpy2DAsync                *cuMemcpy2DAsync;
tcuMemcpy3DAsync                *cuMemcpy3DAsync;
tcuMemsetD8                     *cuMemsetD8;
tcuMemsetD16                    *cuMemsetD16;
tcuMemsetD32                    *cuMemsetD32;
tcuMemsetD2D8                   *cuMemsetD2D8;
tcuMemsetD2D16                  *cuMemsetD2D16;
tcuMemsetD2D32                  *cuMemsetD2D32;
tcuFuncSetBlockShape            *cuFuncSetBlockShape;
tcuFuncSetSharedSize            *cuFuncSetSharedSize;
tcuFuncGetAttribute             *cuFuncGetAttribute;
tcuArrayCreate                  *cuArrayCreate;
tcuArrayGetDescriptor           *cuArrayGetDescriptor;
tcuArrayDestroy                 *cuArrayDestroy;
tcuArray3DCreate                *cuArray3DCreate;
tcuArray3DGetDescriptor         *cuArray3DGetDescriptor;
tcuTexRefCreate                 *cuTexRefCreate;
tcuTexRefDestroy                *cuTexRefDestroy;
tcuTexRefSetArray               *cuTexRefSetArray;
tcuTexRefSetAddress             *cuTexRefSetAddress;
tcuTexRefSetAddress2D           *cuTexRefSetAddress2D;
tcuTexRefSetFormat              *cuTexRefSetFormat;
tcuTexRefSetAddressMode         *cuTexRefSetAddressMode;
tcuTexRefSetFilterMode          *cuTexRefSetFilterMode;
tcuTexRefSetFlags               *cuTexRefSetFlags;
tcuTexRefGetAddress             *cuTexRefGetAddress;
tcuTexRefGetArray               *cuTexRefGetArray;
tcuTexRefGetAddressMode         *cuTexRefGetAddressMode;
tcuTexRefGetFilterMode          *cuTexRefGetFilterMode;
tcuTexRefGetFormat              *cuTexRefGetFormat;
tcuTexRefGetFlags               *cuTexRefGetFlags;
tcuParamSetSize                 *cuParamSetSize;
tcuParamSeti                    *cuParamSeti;
tcuParamSetf                    *cuParamSetf;
tcuParamSetv                    *cuParamSetv;
tcuParamSetTexRef               *cuParamSetTexRef;
tcuLaunch                       *cuLaunch;
tcuLaunchGrid                   *cuLaunchGrid;
tcuLaunchGridAsync              *cuLaunchGridAsync;
tcuEventCreate                  *cuEventCreate;
tcuEventRecord                  *cuEventRecord;
tcuEventQuery                   *cuEventQuery;
tcuEventSynchronize             *cuEventSynchronize;
tcuEventDestroy                 *cuEventDestroy;
tcuEventElapsedTime             *cuEventElapsedTime;
tcuStreamCreate                 *cuStreamCreate;
tcuStreamQuery                  *cuStreamQuery;
tcuStreamSynchronize            *cuStreamSynchronize;
tcuStreamDestroy                *cuStreamDestroy;

#define CHECKED_CALL(call)   result = call; if (CUDA_SUCCESS != result) return result

#if defined(_WIN32) || defined(_WIN64)

    #include <Windows.h>

    #ifdef UNICODE
    static LPCWSTR __CudaLibName = L"nvcuda.dll";
    #else
    static LPCSTR __CudaLibName = "nvcuda.dll";
    #endif

    typedef HMODULE CUDADRIVER;

    CUresult LOAD_LIBRARY(CUDADRIVER *pInstance)
    {
        *pInstance = LoadLibrary(__CudaLibName);
        if (*pInstance == NULL)
        {
            return CUDA_ERROR_UNKNOWN;
        }
        return CUDA_SUCCESS;
    }

    #define GET_PROC_LONG(name, ftype, alias)                       \
        alias = (ftype *)GetProcAddress(CudaDrvLib, #name);         \
        if (alias == NULL) return CUDA_ERROR_UNKNOWN

    #define GET_PROC(name)                                          \
        name = (t##name *)GetProcAddress(CudaDrvLib, #name);        \
        if (name == NULL) return CUDA_ERROR_UNKNOWN

#elif defined(__unix__) || defined(__APPLE__) || defined(__MACOSX)

    #include <dlfcn.h>

    #if defined(__APPLE__) || defined(__MACOSX)
    static char __CudaLibName[] = "/usr/local/cuda/lib/libcuda.dylib";
    #else
    static char __CudaLibName[] = "libcuda.so";
    #endif

    typedef void * CUDADRIVER;

    CUresult LOAD_LIBRARY(CUDADRIVER *pInstance)
    {
        *pInstance = dlopen(__CudaLibName, RTLD_NOW);
        if (*pInstance == NULL)
        {
            return CUDA_ERROR_UNKNOWN;
        }
        return CUDA_SUCCESS;
    }

    #define GET_PROC_LONG(name, ftype, alias)                       \
        alias = (ftype *)dlsym(CudaDrvLib, #name);                  \
        if (alias == NULL) return CUDA_ERROR_UNKNOWN

    #define GET_PROC(name)                                          \
        name = (t##name *)dlsym(CudaDrvLib, #name);                 \
        if (name == NULL) return CUDA_ERROR_UNKNOWN

#endif


CUresult CUDAAPI cuInit(unsigned int Flags)
{
    CUDADRIVER CudaDrvLib;
    CUresult result;
    int driverVer;
    CHECKED_CALL(LOAD_LIBRARY(&CudaDrvLib));

    //cuInit must be present ever
    GET_PROC_LONG(cuInit, tcuInit, _cuInit);
	
    //available since 2.2
    GET_PROC(cuDriverGetVersion);

    //get driver version
    CHECKED_CALL(_cuInit(Flags));
    CHECKED_CALL(cuDriverGetVersion(&driverVer));

    GET_PROC(cuDeviceGet);
    GET_PROC(cuDeviceGetCount);
    GET_PROC(cuDeviceGetName);
    GET_PROC(cuDeviceComputeCapability);
    GET_PROC(cuDeviceTotalMem);
    GET_PROC(cuDeviceGetProperties);
    GET_PROC(cuDeviceGetAttribute);
    GET_PROC(cuCtxCreate);
    GET_PROC(cuCtxDestroy);
    GET_PROC(cuCtxAttach);
    GET_PROC(cuCtxDetach);
    GET_PROC(cuCtxPushCurrent);
    GET_PROC(cuCtxPopCurrent);
    GET_PROC(cuCtxGetDevice);
    GET_PROC(cuCtxSynchronize);
    GET_PROC(cuModuleLoad);
    GET_PROC(cuModuleLoadData);
    GET_PROC(cuModuleLoadDataEx);
    GET_PROC(cuModuleLoadFatBinary);
    GET_PROC(cuModuleUnload);
    GET_PROC(cuModuleGetFunction);
    GET_PROC(cuModuleGetGlobal);
    GET_PROC(cuModuleGetTexRef);
    GET_PROC(cuMemGetInfo);
    GET_PROC(cuMemAlloc);
    GET_PROC(cuMemAllocPitch);
    GET_PROC(cuMemFree);
    GET_PROC(cuMemGetAddressRange);
    GET_PROC(cuMemAllocHost);
    GET_PROC(cuMemFreeHost);
    GET_PROC(cuMemHostAlloc);
    GET_PROC(cuMemHostGetDevicePointer);
    GET_PROC(cuMemcpyHtoD);
    GET_PROC(cuMemcpyDtoH);
    GET_PROC(cuMemcpyDtoD);
    GET_PROC(cuMemcpyDtoA);
    GET_PROC(cuMemcpyAtoD);
    GET_PROC(cuMemcpyHtoA);
    GET_PROC(cuMemcpyAtoH);
    GET_PROC(cuMemcpyAtoA);
    GET_PROC(cuMemcpy2D);
    GET_PROC(cuMemcpy2DUnaligned);
    GET_PROC(cuMemcpy3D);
    GET_PROC(cuMemcpyHtoDAsync);
    GET_PROC(cuMemcpyDtoHAsync);
    GET_PROC(cuMemcpyHtoAAsync);
    GET_PROC(cuMemcpyAtoHAsync);
    GET_PROC(cuMemcpy2DAsync);
    GET_PROC(cuMemcpy3DAsync);
    GET_PROC(cuMemsetD8);
    GET_PROC(cuMemsetD16);
    GET_PROC(cuMemsetD32);
    GET_PROC(cuMemsetD2D8);
    GET_PROC(cuMemsetD2D16);
    GET_PROC(cuMemsetD2D32);
    GET_PROC(cuFuncSetBlockShape);
    GET_PROC(cuFuncSetSharedSize);
    GET_PROC(cuFuncGetAttribute);
    GET_PROC(cuArrayCreate);
    GET_PROC(cuArrayGetDescriptor);
    GET_PROC(cuArrayDestroy);
    GET_PROC(cuArray3DCreate);
    GET_PROC(cuArray3DGetDescriptor);
    GET_PROC(cuTexRefCreate);
    GET_PROC(cuTexRefDestroy);
    GET_PROC(cuTexRefSetArray);
    GET_PROC(cuTexRefSetAddress);
    GET_PROC(cuTexRefSetAddress2D);
    GET_PROC(cuTexRefSetFormat);
    GET_PROC(cuTexRefSetAddressMode);
    GET_PROC(cuTexRefSetFilterMode);
    GET_PROC(cuTexRefSetFlags);
    GET_PROC(cuTexRefGetAddress);
    GET_PROC(cuTexRefGetArray);
    GET_PROC(cuTexRefGetAddressMode);
    GET_PROC(cuTexRefGetFilterMode);
    GET_PROC(cuTexRefGetFormat);
    GET_PROC(cuTexRefGetFlags);
    GET_PROC(cuParamSetSize);
    GET_PROC(cuParamSeti);
    GET_PROC(cuParamSetf);
    GET_PROC(cuParamSetv);
    GET_PROC(cuParamSetTexRef);
    GET_PROC(cuLaunch);
    GET_PROC(cuLaunchGrid);
    GET_PROC(cuLaunchGridAsync);
    GET_PROC(cuEventCreate);
    GET_PROC(cuEventRecord);
    GET_PROC(cuEventQuery);
    GET_PROC(cuEventSynchronize);
    GET_PROC(cuEventDestroy);
    GET_PROC(cuEventElapsedTime);
    GET_PROC(cuStreamCreate);
    GET_PROC(cuStreamQuery);
    GET_PROC(cuStreamSynchronize);
    GET_PROC(cuStreamDestroy);

    if (driverVer >= 2030)
    {
        GET_PROC(cuMemHostGetFlags);
    }

    return CUDA_SUCCESS;
}
