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
 
 #include "particles_kernel.cuh"

extern "C"
{
void initCuda();
void setParameters(SimParams *hostParams);
void createNoiseTexture(int w, int h, int d);

void 
integrateSystem(float4 *oldPos, float4 *newPos,
				float4 *oldVel, float4 *newVel,
                float deltaTime,
                int numParticles);

void 
calcDepth(float4*  pos, 
		  float*   keys,		// output
          uint*    indices,		// output 
          float3   sortVector,
          int      numParticles);
          
}