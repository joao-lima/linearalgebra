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
 
 

typedef struct{
    float x, y;
} Complex;
const Complex CPLX_ZERO = {0, 0};


//a += b * c
void complexMAD(Complex& a, Complex b, Complex c){
    Complex t = {a.x + b.x * c.x - b.y * c.y, a.y + b.x * c.y + b.y * c.x};
    a = t;
}


////////////////////////////////////////////////////////////////////////////////
// Reference straightfroward CPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionCPU(
    Complex *h_Result,
    Complex *h_Data,
    Complex *h_Kernel,
    int dataW,
    int dataH,
    int kernelW,
    int kernelH,
    int kernelX,
    int kernelY
){
    for(int y = 0; y < dataH; y++)
        for(int x = 0; x < dataW; x++){
            Complex sum = CPLX_ZERO;

            for(int ky = -(kernelH - kernelY - 1); ky <= kernelY; ky++)
                for(int kx = -(kernelW - kernelX - 1); kx <= kernelX; kx++){
                    int dx = x + kx;
                    int dy = y + ky;
                    if(dx < 0) dx = 0;
                    if(dy < 0) dy = 0;
                    if(dx >= dataW) dx = dataW - 1;
                    if(dy >= dataH) dy = dataH - 1;

                    complexMAD(
                        sum,
                        h_Data[dy * dataW + dx],
                        h_Kernel[(kernelY - ky) * kernelW + (kernelX - kx)]
                    );
                }

            h_Result[y * dataW + x] = sum;
        }
}
