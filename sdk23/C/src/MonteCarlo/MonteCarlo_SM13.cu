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
#include "MonteCarlo_kernel.cuh"
#include "quasirandomGenerator_kernel.cuh"



extern "C" void initMonteCarlo_SM13(TOptionPlan *plan){
    initMonteCarloGPU(plan);
}

extern "C" void closeMonteCarlo_SM13(TOptionPlan *plan){
    closeMonteCarloGPU(plan);
}

extern "C" void MonteCarlo_SM13(TOptionPlan *plan){
    MonteCarloGPU(plan);
}

extern "C" void inverseCND_SM13(float *d_Output, float *d_Input, unsigned int N){
    inverseCNDgpu(d_Output, d_Input, N);
}
