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

#ifndef __BODYSYSTEMCUDA_H__
#define __BODYSYSTEMCUDA_H__

#include "bodysystem.h"

// CUDA BodySystem: runs on the GPU
class BodySystemCUDA : public BodySystem
{
public:
    BodySystemCUDA(int numBodies, bool usePBO);
    BodySystemCUDA(int numBodies, unsigned int p, unsigned int q, bool usePBO);
    virtual ~BodySystemCUDA();

    virtual void update(float deltaTime);

    virtual void setSoftening(float softening);
    virtual void setDamping(float damping);

    virtual float* getArray(BodyArray array);
    virtual void   setArray(BodyArray array, const float* data);

    virtual unsigned int getCurrentReadBuffer() const 
    { 
        return m_pbo[m_currentRead]; 
    }

    virtual void synchronizeThreads() const;

protected: // methods
    BodySystemCUDA() {}

    virtual void _initialize(int numBodies);
    virtual void _finalize();
    
protected: // data
    // CPU data
    float* m_hPos;
    float* m_hVel;

    // GPU data
    float* m_dPos[2];
    float* m_dVel[2];

    bool m_bUsePBO;

    float m_damping;

    unsigned int m_pbo[2];
    unsigned int m_currentRead;
    unsigned int m_currentWrite;

    unsigned int m_p;
    unsigned int m_q;
};

#endif // __BODYSYSTEMCUDA_H__
