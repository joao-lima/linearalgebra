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



#define DOUBLE_PRECISION
#include "binomialOptions_kernel.cuh"


extern "C" void binomialOptions_SM13(
    float *callValue,
    TOptionData  *optionData,
    int optN
){
    binomialOptionsGPU(callValue, optionData, optN);
}
