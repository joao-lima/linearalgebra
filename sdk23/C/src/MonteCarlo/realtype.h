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



#ifndef REALTYPE_H
#define REALTYPE_H



//Throw out an error for unsupported target
#if defined(DOUBLE_PRECISION) && defined(__CUDACC__) && defined(CUDA_NO_SM_13_DOUBLE_INTRINSICS)
    #error -arch sm_13 nvcc flag is required to compile in double-precision mode
#endif


#ifndef DOUBLE_PRECISION
    typedef float real;
#else
    typedef double real;
#endif



#endif
