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

#ifndef __RENDER_PARTICLES__
#define __RENDER_PARTICLES__

class ParticleRenderer
{
public:
    ParticleRenderer();
    ~ParticleRenderer();

    void setPositions(float *pos, int numParticles);
    void setColors(float *color, int numParticles);
    void setPBO(unsigned int pbo, int numParticles);

    enum DisplayMode
    {
        PARTICLE_POINTS,
        PARTICLE_SPRITES,
        PARTICLE_SPRITES_COLOR,
        PARTICLE_NUM_MODES
    };

    void display(DisplayMode mode = PARTICLE_POINTS);

    void setPointSize(float size)  { m_pointSize = size; }
    void setSpriteSize(float size) { m_spriteSize = size; }

protected: // methods
    void _initGL();
    void _createTexture(int resolution);
    void _drawPoints(bool color = false);

protected: // data
    float *m_pos;
    int m_numParticles;

    float m_pointSize;
    float m_spriteSize;

    unsigned int m_vertexShader;
    unsigned int m_pixelShader;
    unsigned int m_program;
    unsigned int m_texture;
    unsigned int m_pbo;
    unsigned int m_vboColor;
};

#endif //__ RENDER_PARTICLES__
