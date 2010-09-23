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



#include "quasirandomGenerator_kernel.cuh"



extern "C" void initTable_SM10(unsigned int tableCPU[QRNG_DIMENSIONS][QRNG_RESOLUTION]){
    initTableGPU(tableCPU);
}

extern "C" void quasirandomGenerator_SM10(float *d_Output, unsigned int seed, unsigned int N){
    quasirandomGeneratorGPU(d_Output, seed, N);
}

extern "C" void inverseCND_SM10(float *d_Output, float *d_Input, unsigned int N){
    inverseCNDgpu(d_Output, d_Input, N);
}
