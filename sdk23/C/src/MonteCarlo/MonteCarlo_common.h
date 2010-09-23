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



#ifndef MONTECARLO_COMMON_H
#define MONTECARLO_COMMON_H



////////////////////////////////////////////////////////////////////////////////
// Global types
////////////////////////////////////////////////////////////////////////////////
typedef struct{
    float S;
    float X;
    float T;
    float R;
    float V;
} TOptionData;

typedef struct
#ifdef __CUDACC__
__align__(8)
#endif
{
    float Expected;
    float Confidence;
} TOptionValue;


typedef struct{
    //Device ID for multi-GPU version
    int device;
    //Option count for this plan
    int optionCount;

    //Host-side data source and result destination
    TOptionData  *optionData;
    TOptionValue *callValue;
    //Intermediate device-side buffers
    void *d_Buffer;

    //(Pseudo/quasirandom) number generator seed
    unsigned int seed;
    //(Pseudo/quasirandom) samples count
    int pathN;
    //(Pseudo/quasirandom) samples device storage
    float *d_Samples;

    //Time stamp
    float time;
} TOptionPlan;



#endif
