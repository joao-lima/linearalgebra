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
 

#define IMUL(a, b) __mul24(a, b)



////////////////////////////////////////////////////////////////////////////////
// Cyclically shift convolution kernel, so that the center is at (0, 0)
////////////////////////////////////////////////////////////////////////////////
texture<Complex, 2, cudaReadModeElementType> texKernel;

__global__ void padKernel(
    Complex *d_PaddedKernel,
    int fftW,
    int fftH,
    int kernelW,
    int kernelH,
    int kernelX,
    int kernelY
){
    const int x = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;

    if(x < kernelW && y < kernelH){
        int kx = x - kernelX; if(kx < 0) kx += fftW;
        int ky = y - kernelY; if(ky < 0) ky += fftH;
        d_PaddedKernel[IMUL(ky, fftW) + kx] =
            tex2D(texKernel, (float)x + 0.5f, (float)y + 0.5f);
    }
}


////////////////////////////////////////////////////////////////////////////////
// Copy input data array to the upper left corner and pad by border values
////////////////////////////////////////////////////////////////////////////////
texture<Complex, 2, cudaReadModeElementType> texData;

__global__ void padData(
    Complex *d_PaddedData,
    int fftW,
    int fftH,
    int dataW,
    int dataH,
    int kernelW,
    int kernelH,
    int kernelX,
    int kernelY
){
    const int x = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
    const int borderW = dataW + kernelX;
    const int borderH = dataH + kernelY;
    int dx;
    int dy;

    if(x < fftW && y < fftH){
        if(x < dataW) dx = x;
        if(y < dataH) dy = y;
        if(x >= dataW && x < borderW) dx = dataW - 1;
        if(y >= dataH && y < borderH) dy = dataH - 1;
        if(x >= borderW) dx = 0;
        if(y >= borderH) dy = 0;

        d_PaddedData[IMUL(y, fftW) + x] =
            tex2D(texData, (float)dx + 0.5f, (float)dy + 0.5f);
    }
}



////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
__device__ void complexMulAndScale(Complex& a, Complex b, float c){
    Complex t = {c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y)};
    a = t;
}

__global__ void modulateAndNormalize(
    Complex *d_PaddedData,
    Complex *d_PaddedKernel,
    int dataN
){
    const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
    const int threadN = IMUL(blockDim.x, gridDim.x);
    const float     q = 1.0f / (float)dataN;

    for(int i = tid; i < dataN; i += threadN)
        complexMulAndScale(d_PaddedData[i], d_PaddedKernel[i], q);
}
