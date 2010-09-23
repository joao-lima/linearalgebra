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

#ifndef __BODYSYSTEMCPU_H__
#define __BODYSYSTEMCPU_H__

#include "bodysystem.h"

// CPU Body System
class BodySystemCPU : public BodySystem
{
public:
    BodySystemCPU(int numBodies);
    virtual ~BodySystemCPU();

    virtual void update(float deltaTime);

    virtual void setSoftening(float softening) { m_softeningSquared = softening * softening; }
    virtual void setDamping(float damping)     { m_damping = damping; }

    virtual float* getArray(BodyArray array);
    virtual void   setArray(BodyArray array, const float* data);

    virtual unsigned int getCurrentReadBuffer() const { return m_currentRead; }

protected: // methods
    BodySystemCPU() {} // default constructor

    virtual void _initialize(int numBodies);
    virtual void _finalize();

    void _computeNBodyGravitation();
    void _integrateNBodySystem(float deltaTime);
    
protected: // data
    float* m_pos[2];
    float* m_vel[2];
    float* m_force;

    float m_softeningSquared;
    float m_damping;

    unsigned int m_currentRead;
    unsigned int m_currentWrite;
};

#endif // __BODYSYSTEMCPU_H__
